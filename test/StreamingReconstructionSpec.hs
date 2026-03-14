{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleContexts #-}

module StreamingReconstructionSpec (spec) where

import Test.Hspec
import qualified Data.ByteString.Lazy as BSL
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.Encoding as TE
import Data.Aeson (object, (.=), Value)
import qualified Data.Aeson as Aeson
import Data.Default (def)
import Control.Monad (when)
import Data.Maybe (listToMaybe)
import qualified Data.Vector as V
import qualified Data.Aeson.KeyMap as KM
import Network.SSE (parseSSEComplete, sseEventData)
import TestCache (ResponseProvider)
import TestModels
import UniversalLLM
import UniversalLLM.Providers.OpenAI (LlamaCpp(..))
import UniversalLLM.Protocols.OpenAI
import qualified UniversalLLM.Protocols.OpenAI as Proto

-- | Reconstruct a response from SSE streaming deltas.
-- Uses the same SSE parser as the actual streaming interpreter.
reconstructFromSSE :: BSL.ByteString -> Either String OpenAIResponse
reconstructFromSSE sseBody =
  let events = parseSSEComplete (BSL.toStrict sseBody)
      chunks  = [ case Aeson.eitherDecodeStrict (sseEventData ev) of
                    Right val -> Right val
                    Left err  -> Left ("Failed to parse chunk: " ++ err)
                | ev <- events
                , sseEventData ev /= "[DONE]"
                ]
  in case sequence chunks of
       Left err         -> Left err
       Right validChunks ->
         Right $ foldl Proto.mergeOpenAIDelta emptyResponse validChunks
  where
    emptyResponse = OpenAISuccess (OpenAISuccessResponse [])

-- | Compare two OpenAI responses, ignoring certain fields that may differ
compareResponses :: OpenAIResponse -> OpenAIResponse -> Either String ()
compareResponses (OpenAISuccess (OpenAISuccessResponse streamedChoices))
                 (OpenAISuccess (OpenAISuccessResponse nonStreamedChoices)) = do
  -- Check we have the same number of choices
  when (length streamedChoices /= length nonStreamedChoices) $
    Left $ "Different number of choices: streamed has " ++ show (length streamedChoices) ++
           ", non-streamed has " ++ show (length nonStreamedChoices)

  -- Compare each choice
  sequence_ $ zipWith compareChoice streamedChoices nonStreamedChoices

  where
    compareChoice :: OpenAIChoice -> OpenAIChoice -> Either String ()
    compareChoice (OpenAIChoice streamedMsg) (OpenAIChoice nonStreamedMsg) = do
      -- Compare role
      when (role streamedMsg /= role nonStreamedMsg) $
        Left $ "Different roles: streamed=" ++ T.unpack (role streamedMsg) ++
               ", non-streamed=" ++ T.unpack (role nonStreamedMsg)

      -- Compare content (if present)
      when (content streamedMsg /= content nonStreamedMsg) $
        Left $ "Different content:\nStreamed: " ++ show (content streamedMsg) ++
               "\nNon-streamed: " ++ show (content nonStreamedMsg)

      -- Compare reasoning_content (if present)
      when (reasoning_content streamedMsg /= reasoning_content nonStreamedMsg) $
        Left $ "Different reasoning_content:\nStreamed: " ++ show (reasoning_content streamedMsg) ++
               "\nNon-streamed: " ++ show (reasoning_content nonStreamedMsg)

      -- Compare tool_calls (if present)
      -- Note: Tool calls are the most complex to reconstruct
      case (tool_calls streamedMsg, tool_calls nonStreamedMsg) of
        (Just streamedCalls, Just nonStreamedCalls) -> compareToolCalls streamedCalls nonStreamedCalls
        (Nothing, Nothing) -> Right ()
        (Just _, Nothing) -> Left "Streamed has tool_calls but non-streamed doesn't"
        (Nothing, Just _) -> Left "Non-streamed has tool_calls but streamed doesn't"

      return ()

    compareToolCalls :: [OpenAIToolCall] -> [OpenAIToolCall] -> Either String ()
    compareToolCalls streamedCalls nonStreamedCalls = do
      when (length streamedCalls /= length nonStreamedCalls) $
        Left $ "Different number of tool calls: streamed has " ++ show (length streamedCalls) ++
               ", non-streamed has " ++ show (length nonStreamedCalls)

      sequence_ $ zipWith compareToolCall streamedCalls nonStreamedCalls

      return ()

    compareToolCall :: OpenAIToolCall -> OpenAIToolCall -> Either String ()
    compareToolCall streamed nonStreamed = do
      -- Note: We skip comparing callId because llamacpp doesn't respect seed for ID generation
      -- The IDs are randomly generated and will differ between requests even with the same seed

      -- Compare function name
      let streamedName = toolFunctionName (toolFunction streamed)
          nonStreamedName = toolFunctionName (toolFunction nonStreamed)
      when (streamedName /= nonStreamedName) $
        Left $ "Different function names: streamed=" ++ T.unpack streamedName ++
               ", non-streamed=" ++ T.unpack nonStreamedName

      -- Compare function arguments (as JSON to handle different whitespace)
      let streamedArgs = toolFunctionArguments (toolFunction streamed)
          nonStreamedArgs = toolFunctionArguments (toolFunction nonStreamed)

      -- Try to parse and compare as JSON
      case (Aeson.eitherDecodeStrict (TE.encodeUtf8 streamedArgs) :: Either String Value,
            Aeson.eitherDecodeStrict (TE.encodeUtf8 nonStreamedArgs) :: Either String Value) of
        (Right streamedJSON, Right nonStreamedJSON) ->
          when (streamedJSON /= nonStreamedJSON) $
            Left $ "Different function arguments (as JSON):\nStreamed: " ++ show streamedJSON ++
                   "\nNon-streamed: " ++ show nonStreamedJSON
        (Left err, _) ->
          Left $ "Failed to parse streamed arguments as JSON: " ++ err
        (_, Left err) ->
          Left $ "Failed to parse non-streamed arguments as JSON: " ++ err

      return ()

compareResponses (OpenAIError err1) (OpenAIError err2) =
  if err1 == err2
    then Right ()
    else Left $ "Different errors:\nStreamed: " ++ show err1 ++ "\nNon-streamed: " ++ show err2
compareResponses (OpenAISuccess _) (OpenAIError err) =
  Left $ "Streamed succeeded but non-streamed failed: " ++ show err
compareResponses (OpenAIError err) (OpenAISuccess _) =
  Left $ "Streamed failed but non-streamed succeeded: " ++ show err

spec :: Maybe [String]  -- Loaded model variants (for llamacpp)
     -> ResponseProvider OpenAIRequest OpenAIResponse
     -> ResponseProvider OpenAIRequest BSL.ByteString
     -> Spec
spec loadedModel getNonStreamingResponse getStreamingResponse = do
  describe "Streaming Reconstruction Tests" $ do

    it "reconstructs simple text response correctly" $ do
      let seed = 42
          configs = [MaxTokens 300, Seed seed]
          msgs = [UserText "Say hello in one sentence. Keep thinking to absolute minimum."]

          -- Build non-streaming request using GLM45Air as a placeholder model
          -- The actual model loaded in llamacpp is handled by the modelMatches filter
          model = Model GLM45Air LlamaCpp
          provider = llamaCppGLM45
          state = (def, ((), ((), ())))
          nonStreamReq = snd $ toProviderRequest provider model configs state msgs

          -- Build streaming request
          streamReq = Proto.enableOpenAIStreaming nonStreamReq

      -- Verify seed is set
      Proto.seed nonStreamReq `shouldBe` Just seed
      Proto.seed streamReq `shouldBe` Just seed
      Proto.stream streamReq `shouldBe` Just True

      -- Get non-streaming response
      nonStreamResp <- getNonStreamingResponse nonStreamReq

      -- Get streaming response and reconstruct
      sseBody <- getStreamingResponse streamReq

      case reconstructFromSSE sseBody of
        Left err -> expectationFailure $ "Failed to reconstruct SSE response: " ++ err
        Right streamResp ->
          case compareResponses streamResp nonStreamResp of
            Left diff -> expectationFailure $ "Reconstructed response differs from non-streamed:\n" ++ diff
            Right () -> return ()

    it "reconstructs tool call response correctly" $ do
      let seed = 123
          toolDef = ToolDefinition
            { toolDefName = "get_weather"
            , toolDefDescription = "Get the weather for a location"
            , toolDefParameters = object
                [ "type" .= ("object" :: Text)
                , "properties" .= object
                    [ "location" .= object
                        [ "type" .= ("string" :: Text)
                        , "description" .= ("City name" :: Text)
                        ]
                    ]
                , "required" .= (["location"] :: [Text])
                ]
            }
          configs = [MaxTokens 300, Tools [toolDef], Seed seed]
          msgs = [UserText "What's the weather in Paris? Keep thinking to absolute minimum, just call the tool."]

          model = Model GLM45Air LlamaCpp
          provider = llamaCppGLM45
          state = (def, ((), ((), ())))
          nonStreamReq = snd $ toProviderRequest provider model configs state msgs
          streamReq = Proto.enableOpenAIStreaming nonStreamReq

      -- Get both responses
      nonStreamResp <- getNonStreamingResponse nonStreamReq
      sseBody <- getStreamingResponse streamReq

      case reconstructFromSSE sseBody of
        Left err -> expectationFailure $ "Failed to reconstruct SSE response with tools: " ++ err
        Right streamResp ->
          case compareResponses streamResp nonStreamResp of
            Left diff -> expectationFailure $ "Tool call reconstruction differs:\n" ++ diff
            Right () -> return ()

    it "reconstructs reasoning response correctly" $ do
      let seed = 456
          configs = [MaxTokens 300, Reasoning True, Seed seed]
          msgs = [UserText "What is 7 + 5? Keep thinking to absolute minimum, answer immediately."]

          model = Model GLM45Air LlamaCpp
          provider = llamaCppGLM45
          state = (def, ((), ((), ())))
          nonStreamReq = snd $ toProviderRequest provider model configs state msgs
          streamReq = Proto.enableOpenAIStreaming nonStreamReq

      -- Get both responses
      nonStreamResp <- getNonStreamingResponse nonStreamReq
      sseBody <- getStreamingResponse streamReq

      case reconstructFromSSE sseBody of
        Left err -> expectationFailure $ "Failed to reconstruct SSE response with reasoning: " ++ err
        Right streamResp ->
          case compareResponses streamResp nonStreamResp of
            Left diff -> expectationFailure $ "Reasoning reconstruction differs:\n" ++ diff
            Right () -> return ()
