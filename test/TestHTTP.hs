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
import System.Environment (lookupEnv)
import System.IO (hPutStrLn, stdout)

-- Generic HTTP POST call that encodes request and decodes response via autodocodec
-- Returns (status code, decoded response)
httpCall :: (HasCodec req, HasCodec resp)
         => String              -- ^ Endpoint URL
         -> [(T.Text, T.Text)]  -- ^ Headers
         -> req
         -> IO (Int, resp)
httpCall endpoint headers request = do
  debug <- lookupEnv "DEBUG_HTTP"
  req <- parseRequest $ "POST " ++ endpoint

  let body = Aeson.encode $ toJSONViaCodec request
      headerList = [(CI.mk $ TE.encodeUtf8 k, TE.encodeUtf8 v) | (k, v) <- headers]
  let req' = setRequestHeaders headerList
           $ setRequestBodyLBS body
           $ setRequestResponseTimeout (responseTimeoutMicro 300000000)  -- 5 minutes timeout
           $ req

  case debug of
    Just _ -> do
      hPutStrLn stdout $ "[DEBUG_HTTP] --> POST " ++ endpoint
      BSL.hPut stdout body >> hPutStrLn stdout ""
    Nothing -> return ()

  response <- httpLBS req'
  let responseBody = getResponseBody response
      statusCode = getResponseStatusCode response

  case debug of
    Just _ -> do
      hPutStrLn stdout $ "[DEBUG_HTTP] <-- " ++ show statusCode
      BSL.hPut stdout responseBody >> hPutStrLn stdout ""
    Nothing -> return ()
  case eitherDecodeJSONViaCodec responseBody of
    Right resp -> return (statusCode, resp)
    Left err -> error $ "Failed to parse response: " ++ err ++ "\nResponse body: " ++ show responseBody

-- HTTP POST call for streaming responses that returns raw response body as ByteString
-- Used for SSE (Server-Sent Events) responses which can't be parsed as JSON
-- Returns (status code, response body)
httpCallStreaming :: (HasCodec req)
                  => String              -- ^ Endpoint URL
                  -> [(T.Text, T.Text)]  -- ^ Headers
                  -> req
                  -> IO (Int, BSL.ByteString)
httpCallStreaming endpoint headers request = do
  req <- parseRequest $ "POST " ++ endpoint

  let headerList = [(CI.mk $ TE.encodeUtf8 k, TE.encodeUtf8 v) | (k, v) <- headers]
  let req' = setRequestHeaders headerList
           $ setRequestBodyLBS (Aeson.encode $ toJSONViaCodec request)
           $ setRequestResponseTimeout (responseTimeoutMicro 300000000)  -- 5 minutes timeout
           $ req

  response <- httpLBS req'
  let statusCode = getResponseStatusCode response
      responseBody = getResponseBody response
  return (statusCode, responseBody)

-- HTTP GET call for raw JSON responses
httpGet :: String -> IO BSL.ByteString
httpGet endpoint = do
  req <- parseRequest $ "GET " ++ endpoint
  response <- httpLBS req
  return $ getResponseBody response
