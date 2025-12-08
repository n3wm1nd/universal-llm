{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}

module TestHTTP
  ( httpCall
  , httpCallStreaming
  , httpGet
  ) where

import qualified Data.Text as T
import qualified Data.Text.Encoding as TE
import qualified Data.ByteString.Lazy as BSL
import qualified Data.Aeson as Aeson
import Autodocodec (toJSONViaCodec, eitherDecodeJSONViaCodec, HasCodec)
import Network.HTTP.Simple
import Network.HTTP.Client.Conduit (responseTimeoutMicro)
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
           $ setRequestBodyLBS (Aeson.encode $ toJSONViaCodec request)
           $ setRequestResponseTimeout (responseTimeoutMicro 300000000)  -- 5 minutes timeout
           $ req

  response <- httpLBS req'
  let responseBody = getResponseBody response
  case eitherDecodeJSONViaCodec responseBody of
    Right resp -> return resp
    Left err -> error $ "Failed to parse response: " ++ err ++ "\nResponse body: " ++ show responseBody

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
           $ setRequestBodyLBS (Aeson.encode $ toJSONViaCodec request)
           $ setRequestResponseTimeout (responseTimeoutMicro 300000000)  -- 5 minutes timeout
           $ req

  response <- httpLBS req'
  return $ getResponseBody response

-- HTTP GET call for raw JSON responses
httpGet :: String -> IO BSL.ByteString
httpGet endpoint = do
  req <- parseRequest $ "GET " ++ endpoint
  response <- httpLBS req
  return $ getResponseBody response
