{-# LANGUAGE OverloadedStrings #-}

module ErrorClassifier
  ( transientByStatus
  , transientOpenAI
  , transientAnthropic
  ) where

import qualified Data.Text as T
import TestCache (IsTransient)
import UniversalLLM.Protocols.OpenAI (OpenAIResponse(..), OpenAIErrorResponse(..), OpenAIErrorDetail(..))
import qualified UniversalLLM.Protocols.OpenAI as OpenAI
import UniversalLLM.Protocols.Anthropic (AnthropicResponse(..))

-- | Transient if status is 429, 401, 403, or 5xx.
transientByStatus :: IsTransient resp
transientByStatus statusCode _ =
  statusCode == 429
  || statusCode == 401
  || statusCode == 403
  || (statusCode >= 500 && statusCode < 600)

-- | Transient for OpenAI-format responses: uses error type field when present
-- for finer-grained classification, falls back to HTTP status.
transientOpenAI :: IsTransient OpenAIResponse
transientOpenAI statusCode response = case response of
  OpenAISuccess _ -> False
  OpenAIError (OpenAIErrorResponse errDetail) ->
    case OpenAI.errorType errDetail of
      Just errType
        | "rate_limit"          `T.isInfixOf` T.toLower errType -> True
        | "insufficient_quota"  `T.isInfixOf` T.toLower errType -> True
        | "quota_exceeded"      `T.isInfixOf` T.toLower errType -> True
        | "server_error"        `T.isInfixOf` T.toLower errType -> True
        | "service_unavailable" `T.isInfixOf` T.toLower errType -> True
        | "timeout"             `T.isInfixOf` T.toLower errType -> True
        | "authentication"      `T.isInfixOf` T.toLower errType -> True
        | "unauthorized"        `T.isInfixOf` T.toLower errType -> True
        | "permission"          `T.isInfixOf` T.toLower errType -> True
        | otherwise -> False
      Nothing -> transientByStatus statusCode response

-- | Transient for Anthropic-format responses: falls back to HTTP status.
transientAnthropic :: IsTransient AnthropicResponse
transientAnthropic statusCode response = transientByStatus statusCode response
