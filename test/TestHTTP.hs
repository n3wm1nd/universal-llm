{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}

module TestHTTP
  ( httpCall
  , httpCallStreaming
  ) where

import qualified Data.Text as T
import qualified Data.Text.Encoding as TE
import qualified Data.ByteString.Lazy as BSL
import qualified Data.Aeson as Aeson
import Autodocodec (toJSONViaCodec, eitherDecodeJSONViaCodec, HasCodec)
import Network.HTTP.Simple
import qualified Data.CaseInsensitive as CI

-- Generic HTTP POST call that encodes request and decodes response via autodocodec
httpCall :: (HasCodec req, HasCodec resp)
         => String              -- ^ Endpoint URL
         -> [(T.Text, T.Text)]  -- ^ Headers
         -> req
         -> IO resp
httpCall endpoint headers request = do
  req <- parseRequest $ "POST " ++ endpoint

  let headerList = [(CI.mk $ TE.encodeUtf8 k, TE.encodeUtf8 v) | (k, v) <- headers]
  let req' = setRequestHeaders headerList
           $ setRequestBodyLBS (Aeson.encode $ toJSONViaCodec request) req

  response <- httpLBS req'
  case eitherDecodeJSONViaCodec (getResponseBody response) of
    Right resp -> return resp
    Left err -> error $ "Failed to parse response: " ++ err

-- HTTP POST call for streaming responses that returns raw response body as ByteString
-- Used for SSE (Server-Sent Events) responses which can't be parsed as JSON
httpCallStreaming :: (HasCodec req)
                  => String              -- ^ Endpoint URL
                  -> [(T.Text, T.Text)]  -- ^ Headers
                  -> req
                  -> IO BSL.ByteString
httpCallStreaming endpoint headers request = do
  req <- parseRequest $ "POST " ++ endpoint

  let headerList = [(CI.mk $ TE.encodeUtf8 k, TE.encodeUtf8 v) | (k, v) <- headers]
  let req' = setRequestHeaders headerList
           $ setRequestBodyLBS (Aeson.encode $ toJSONViaCodec request) req

  response <- httpLBS req'
  return $ getResponseBody response
