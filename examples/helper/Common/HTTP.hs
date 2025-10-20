{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE RankNTypes #-}

module Common.HTTP
  ( LLMCall
  , mkLLMCall
  ) where

import UniversalLLM.Core.Types
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.Encoding as TE
import qualified Data.Aeson as Aeson
import Autodocodec (toJSONViaCodec, eitherDecodeJSONViaCodec, HasCodec)
import Network.HTTP.Simple
import qualified Data.CaseInsensitive as CI
import Control.Monad.Trans.Except (ExceptT, withExceptT, except)
import Control.Monad.IO.Class (liftIO)

-- | Type alias for LLM API call function
-- A function that takes a request and returns a response in ExceptT
type LLMCall request response = request -> ExceptT LLMError IO response

-- | Build an LLM call function with endpoint and headers
-- Returns a partially applied function that can be used to make API calls
mkLLMCall :: (HasCodec request, HasCodec response)
          => String           -- ^ API endpoint URL
          -> [(Text, Text)]   -- ^ HTTP headers
          -> LLMCall request response
mkLLMCall endpoint headers = \request -> do
  req <- liftIO $ parseRequest $ "POST " ++ endpoint

  -- Convert headers from [(Text, Text)] to proper header format
  let headerList = [(CI.mk $ TE.encodeUtf8 k, TE.encodeUtf8 v) | (k, v) <- headers]
  let req' = setRequestHeaders headerList
           $ setRequestBodyLBS (Aeson.encode $ toJSONViaCodec request) req

  response <- httpLBS req'
  withExceptT (ParseError . T.pack) $ except $ eitherDecodeJSONViaCodec (getResponseBody response)
