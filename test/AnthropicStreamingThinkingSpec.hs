{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}

module AnthropicStreamingThinkingSpec (spec) where

import Test.Hspec
import qualified Data.ByteString.Lazy as BSL
import Data.Aeson (object, (.=))
import qualified Data.Aeson as Aeson
import qualified Data.Aeson.KeyMap as KeyMap
import UniversalLLM
import UniversalLLM.Protocols.Anthropic
import qualified UniversalLLM.Protocols.Anthropic as Proto
import qualified UniversalLLM.Providers.Anthropic as Provider
import TestModels
import Data.Default (def)

-- Test that thinking block signatures are preserved through streaming reassembly
spec :: Spec
spec = describe "Anthropic Streaming Thinking Signatures" $ do

  it "preserves thinking block signature through SSE reassembly" $ do
    -- REAL SSE response format from Anthropic (based on cached test responses)
    -- Signature starts as empty string and is filled via signature_delta events
    let sseResponse = BSL.intercalate "\n"
          [ "event: message_start"
          , "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_123\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":\"claude-sonnet-4.5-20250929\",\"stop_reason\":null,\"usage\":{\"input_tokens\":100,\"output_tokens\":0}}}"
          , ""
          , "event: content_block_start"
          , "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\",\"signature\":\"\"}}"
          , ""
          , "event: content_block_delta"
          , "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"ErsDCkYICRgCKkDJJNN8UCvrer\"}}"
          , ""
          , "event: content_block_delta"
          , "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"I need to call getcwd\"}}"
          , ""
          , "event: content_block_stop"
          , "data: {\"type\":\"content_block_stop\",\"index\":0}"
          , ""
          , "event: content_block_start"
          , "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_123\",\"name\":\"getcwd\"}}"
          , ""
          , "event: content_block_delta"
          , "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{}\"}}"
          , ""
          , "event: content_block_stop"
          , "data: {\"type\":\"content_block_stop\",\"index\":1}"
          , ""
          , "event: message_delta"
          , "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"}}"
          , ""
          , "event: message_stop"
          , "data: {\"type\":\"message_stop\"}"
          , ""
          ]

    -- Reassemble the SSE into an AnthropicResponse
    let emptyResp = AnthropicSuccessResponse "" "" "assistant" [] Nothing (AnthropicUsage 0 0)
        chunks = parseSSE sseResponse
        finalResp = foldl mergeAnthropicDelta (AnthropicSuccess emptyResp) chunks

    case finalResp of
      AnthropicError err -> expectationFailure $ "Failed to reassemble: " ++ show err
      AnthropicSuccess resp -> do
        let blocks = responseContent resp

        -- Should have thinking block and tool call
        length blocks `shouldBe` 2

        case blocks of
          [AnthropicThinkingBlock thinking sig _, AnthropicToolUseBlock toolId toolName toolInput _] -> do
            thinking `shouldBe` "I need to call getcwd"
            -- Signature should be the accumulated base64 string
            sig `shouldBe` Aeson.String "ErsDCkYICRgCKkDJJNN8UCvrer"
            toolId `shouldBe` "toolu_123"
            toolName `shouldBe` "getcwd"
            toolInput `shouldBe` object []  -- Empty object for getcwd

            -- Now test that we can build a second request with this thinking block
            let model = Model ClaudeSonnet45WithReasoning Provider.Anthropic
                cp = anthropicSonnet45Reasoning
                initialState = (def, ((), ()))

            -- Parse the response to get Message types
            let (state1, msgs1) = either (error . show) id $
                  fromProviderResponse cp model [] initialState finalResp

            -- msgs1 should contain AssistantReasoning and AssistantTool
            length msgs1 `shouldSatisfy` (>= 2)

            -- Build a second request with tool result
            let toolResult = ToolResultMsg (ToolResult (ToolCall toolId toolName toolInput) (Right (object ["result" .= ("/home/user" :: String)])))
                msgs2 = [UserText "test"] ++ msgs1 ++ [toolResult]
                (state2, req2) = toProviderRequest cp model [Reasoning True] state1 msgs2

            -- The request should have thinking config enabled (required when sending thinking blocks)
            Proto.thinking req2 `shouldNotBe` Nothing

            -- Check that the thinking block has a valid signature
            let reqMsgs = messages req2
            reqMsgs `shouldSatisfy` (not . null)

            -- Extract thinking blocks from request messages
            let allBlocks = concatMap content reqMsgs
                thinkingBlocks = [b | b@(AnthropicThinkingBlock _ _ _) <- allBlocks]
            length thinkingBlocks `shouldSatisfy` (> 0)

            -- Verify the signature is preserved correctly (should be a string in this case)
            case thinkingBlocks of
              (AnthropicThinkingBlock _ sig _ : _) -> do
                -- Signature should be present and non-empty
                sig `shouldNotBe` Aeson.String ""
                sig `shouldNotBe` Aeson.Null
              _ -> expectationFailure "No thinking blocks found"

          _ -> expectationFailure $ "Unexpected blocks: " ++ show blocks

-- Import parseSSE helper
parseSSE :: BSL.ByteString -> [Aeson.Value]
parseSSE body =
    let lines = BSL.split (fromIntegral $ fromEnum '\n') body
        dataLines = [l | l <- lines, BSL.isPrefixOf "data: " l]
        jsonStrings = [BSL.drop 6 line | line <- dataLines]
        values = [v | json <- jsonStrings, Just v <- [Aeson.decode json]]
    in values
