{-# LANGUAGE OverloadedStrings #-}

module ErrorClassifier
  ( classifyOpenAIError
  , classifyAnthropicError
  ) where

import qualified Data.Text as T
import UniversalLLM.Protocols.OpenAI (OpenAIResponse(..), OpenAIErrorResponse(..), OpenAIErrorDetail(..))
import qualified UniversalLLM.Protocols.OpenAI as OpenAI
import UniversalLLM.Protocols.Anthropic (AnthropicResponse(..), AnthropicErrorResponse(..))
import qualified UniversalLLM.Protocols.Anthropic as Anthropic

-- | Classify OpenAI-format errors (OpenAI, OpenRouter, etc.)
-- Returns Just msg if error is transient (should not be cached, mark test as pending)
-- Returns Nothing if error should be cached (permanent error from our bad code)
classifyOpenAIError :: Int -> OpenAIResponse -> Maybe String
classifyOpenAIError statusCode response = case response of
  OpenAISuccess _ -> Nothing  -- Not an error
  OpenAIError (OpenAIErrorResponse errDetail) ->
    let code = OpenAI.code errDetail
        msg = OpenAI.errorMessage errDetail
    in classifyByHTTPStatus statusCode msg

-- | Classify Anthropic-format errors
classifyAnthropicError :: Int -> AnthropicResponse -> Maybe String
classifyAnthropicError statusCode response = case response of
  AnthropicSuccess _ -> Nothing  -- Not an error
  AnthropicError errResp ->
    let msg = Anthropic.errorMessage errResp
    in classifyByHTTPStatus statusCode msg

-- | Conservative error classification based on HTTP status codes
-- When in doubt: DON'T CACHE (return Just with pending message)
classifyByHTTPStatus :: Int -> T.Text -> Maybe String
classifyByHTTPStatus statusCode msg
  -- 429 - Rate limit (definitely transient)
  | statusCode == 429 = Just $ "Transient error (429): " ++ T.unpack (T.take 150 msg)

  -- 5xx - Server errors (don't cache - could be transient)
  | statusCode >= 500 && statusCode < 600 =
      Just $ "Transient error (" ++ show statusCode ++ "): " ++ T.unpack (T.take 150 msg)

  -- 401/403 - Auth errors (user's wrong API key, don't cache)
  | statusCode == 401 = Just $ "Transient error (401): " ++ T.unpack (T.take 150 msg)
  | statusCode == 403 = Just $ "Transient error (403): " ++ T.unpack (T.take 150 msg)

  -- 400 - Bad request (our fault, cache it)
  | statusCode == 400 = Nothing

  -- For any other status code, don't cache (conservative)
  | statusCode /= 200 = Just $ "Transient error (" ++ show statusCode ++ "): " ++ T.unpack (T.take 150 msg)

  -- HTTP 200 with error response - be conservative, don't cache unless we add specific rules
  | otherwise = Just $ "Transient error (200): " ++ T.unpack (T.take 150 msg)
