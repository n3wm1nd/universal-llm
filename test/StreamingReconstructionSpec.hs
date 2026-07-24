{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleContexts #-}

module StreamingReconstructionSpec (spec) where

import Test.Hspec
import qualified Data.ByteString.Lazy as BSL
import qualified Data.ByteString.Char8 as BS8
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
import TestCache (ResponseProvider, request)
import TestModels
import UniversalLLM
import UniversalLLM.Providers.OpenAI (LlamaCpp(..))
import UniversalLLM.Protocols.OpenAI
import qualified UniversalLLM.Protocols.OpenAI as Proto
import UniversalLLM.Protocols.OpenAI.Delta (Delta(..))

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
    compareChoice (OpenAIChoice streamedMsg _) (OpenAIChoice nonStreamedMsg _) = do
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
          state = def
          nonStreamReq = snd $ toProviderRequest provider model configs state msgs

          -- Build streaming request
          streamReq = Proto.enableOpenAIStreaming nonStreamReq

      -- Verify seed is set
      Proto.seed nonStreamReq `shouldBe` Just seed
      Proto.seed streamReq `shouldBe` Just seed
      Proto.stream streamReq `shouldBe` Just True

      -- Get non-streaming response
      nonStreamResp <- request getNonStreamingResponse nonStreamReq

      -- Get streaming response and reconstruct
      sseBody <- request getStreamingResponse streamReq

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
          state = def
          nonStreamReq = snd $ toProviderRequest provider model configs state msgs
          streamReq = Proto.enableOpenAIStreaming nonStreamReq

      -- Get both responses
      nonStreamResp <- request getNonStreamingResponse nonStreamReq
      sseBody <- request getStreamingResponse streamReq

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
          state = def
          nonStreamReq = snd $ toProviderRequest provider model configs state msgs
          streamReq = Proto.enableOpenAIStreaming nonStreamReq

      -- Get both responses
      nonStreamResp <- request getNonStreamingResponse nonStreamReq
      sseBody <- request getStreamingResponse streamReq

      case reconstructFromSSE sseBody of
        Left err -> expectationFailure $ "Failed to reconstruct SSE response with reasoning: " ++ err
        Right streamResp ->
          case compareResponses streamResp nonStreamResp of
            Left diff -> expectationFailure $ "Reasoning reconstruction differs:\n" ++ diff
            Right () -> return ()

    -- llama.cpp-only: emits periodic top-level `prompt_progress` chunks
    -- during prefill when the request carries `return_progress: true`. This
    -- has no non-streaming counterpart (prefill has already finished by the
    -- time a non-streaming response comes back), so unlike the tests above
    -- there is nothing to reconstruct-and-compare here — we just assert the
    -- streamed chunks decode into monotonically increasing
    -- 'StreamingPromptProgress' events bounded by the reported total.
    --
    -- Confirmed on a live llama.cpp server (b10088): the plain OpenAI-shaped
    -- request produces *no* progress/heartbeat events at all during prefill
    -- (SSE is silent until generation starts); `return_progress: true` is
    -- required to opt in.
    it "emits prompt_progress events during prefill when return_progress is enabled" $ do
      let seed = 789
          -- A long, unique-but-deterministic filler prompt: long enough that
          -- prefill takes multiple seconds (llama.cpp reports progress on an
          -- interval), unique enough it won't already sit in the server's
          -- prompt cache on a fresh run against a live server.
          filler = T.unwords
            [ "token" <> T.pack (show (i :: Int)) <> "_" <> T.pack (show seed)
            | i <- [1 .. 3000 :: Int]
            ]
          configs = [MaxTokens 10, Seed seed]
          msgs = [UserText (filler <> " Now, in one short sentence, say hello.")]

          model = Model GLM45Air LlamaCpp
          provider = llamaCppGLM45
          state = def
          baseReq = snd $ toProviderRequest provider model configs state msgs
          streamReq = Proto.enableLlamaCppPromptProgress (Proto.enableOpenAIStreaming baseReq)

      Proto.return_progress streamReq `shouldBe` Just True

      sseBody <- request getStreamingResponse streamReq

      let events = parseSSEComplete (BSL.toStrict sseBody)
          chunkValues =
            [ val
            | ev <- events
            , sseEventData ev /= "[DONE]"
            , Right val <- [Aeson.eitherDecodeStrict (sseEventData ev) :: Either String Value]
            ]
          progressEvents =
            [ (processed, total)
            | val <- chunkValues
            , delta <- [Delta val]
            , content <- extractStreamingContent @OpenAIResponse delta
            , StreamingPromptProgress processed total <- [content]
            ]

      -- The filler prompt is long enough that prefill spans multiple
      -- reporting intervals, so at least one prompt_progress event is
      -- required here, not just tolerated -- an empty list means either the
      -- server stopped emitting them or extractStreamingContent regressed.
      progressEvents `shouldNotBe` []

      let totals = map snd progressEvents
          processedCounts = map fst progressEvents
      totals `shouldSatisfy` all (== head totals)
      processedCounts `shouldSatisfy` \ps -> and (zipWith (<=) ps (tail ps))
      last processedCounts `shouldSatisfy` (<= head totals)

    -- Pure decoder test pinned to the exact chunk shape captured from a live
    -- llama.cpp server (b10088) with return_progress: true -- independent of
    -- server/network/timing, so it can't be skipped by a fast/cached prefill.
    it "decodes a literal llama.cpp prompt_progress chunk" $ do
      let raw :: BS8.ByteString
          raw = "{\"choices\":[{\"finish_reason\":null,\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":null}}],\
                \\"created\":1784866039,\"id\":\"chatcmpl-7eYiMcyUdtXF0UeTN7Q5Fri1NpzyKfIh\",\
                \\"model\":\"/models/Laguna-S-2.1-UD-Q4_K_XL-00001-of-00003.gguf\",\
                \\"system_fingerprint\":\"b10088-67b9b0e7f\",\"object\":\"chat.completion.chunk\",\
                \\"timings\":{\"cache_n\":0,\"prompt_n\":6178,\"prompt_ms\":5374.535,\
                \\"prompt_per_token_ms\":0.8699473939786339,\"prompt_per_second\":1149.494793503066,\
                \\"predicted_n\":281,\"predicted_ms\":19218.661,\"predicted_per_token_ms\":68.39381138790036,\
                \\"predicted_per_second\":14.621205920641403},\
                \\"prompt_progress\":{\"total\":49740,\"cache\":0,\"processed\":6178,\"time_ms\":11545}}"

      case Aeson.eitherDecodeStrict raw of
        Left err -> expectationFailure $ "Failed to parse literal chunk: " ++ err
        Right val ->
          extractStreamingContent @OpenAIResponse (Delta val)
            `shouldBe` [StreamingPromptProgress 6178 49740]
