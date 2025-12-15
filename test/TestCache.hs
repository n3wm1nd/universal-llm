{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RankNTypes #-}

module TestCache
  ( CachePath
  , ResponseProvider
  , recordResponse
  , lookupResponse
  , cachedRequest
  , recordMode
  , updateMode
  , playbackMode
  , liveMode
  , recordRawResponse
  , lookupRawResponse
  , cachedRawRequest
  , recordModeRaw
  , updateModeRaw
  , playbackModeRaw
  , CacheMissException(..)
  ) where

import qualified Data.ByteString as BS
import qualified Data.ByteString.Lazy as BSL
import qualified Data.ByteString.Char8 as BS8
import qualified Crypto.Hash.SHA256 as SHA256
import qualified Data.ByteString.Base16 as Base16
import System.Directory (doesDirectoryExist, doesFileExist)
import System.FilePath ((</>))
import Control.Monad (unless)
import Control.Exception (Exception, throwIO, catch)
import Autodocodec (HasCodec, toJSONViaCodec, parseJSONViaCodec)
import Data.Aeson.Types (parseEither)
import qualified Data.Aeson as Aeson
import Test.Hspec (pendingWith)

-- | Exception thrown when a cache miss occurs in playback mode
data CacheMissException = CacheMissException String
  deriving (Show)

instance Exception CacheMissException

type CachePath = FilePath

-- Response provider: abstracts how responses are obtained (cache/live/playback)
-- Parameterized by request and response types
type ResponseProvider req resp = req -> IO resp

-- Hash a request to get cache key
hashRequest :: HasCodec req => req -> String
hashRequest req =
  let jsonBytes = BSL.toStrict $ Aeson.encode $ toJSONViaCodec req
      hash = SHA256.hash jsonBytes
      hashHex = Base16.encode hash
  in BS8.unpack hashHex

-- Record a response for a given request
-- Requires cache directory to already exist
-- Also saves the request for debugging purposes
recordResponse :: (HasCodec req, HasCodec resp) => CachePath -> req -> resp -> IO ()
recordResponse cachePath req resp = do
  dirExists <- doesDirectoryExist cachePath
  unless dirExists $ error $ "Cache directory does not exist: " ++ cachePath
  let cacheKey = hashRequest req
      responsePath = cachePath </> cacheKey <> ".json"
      requestPath = cachePath </> cacheKey <> "-request.json"
      responseJson = Aeson.encode $ toJSONViaCodec resp
      requestJson = Aeson.encode $ toJSONViaCodec req
  BSL.writeFile responsePath responseJson
  BSL.writeFile requestPath requestJson

-- Look up a cached response for a request
lookupResponse :: (HasCodec req, HasCodec resp) => CachePath -> req -> IO (Maybe resp)
lookupResponse cachePath req = do
  let cacheKey = hashRequest req
      cachePath' = cachePath </> cacheKey <> ".json"
  exists <- doesFileExist cachePath'
  if exists
    then do
      contents <- BSL.readFile cachePath'
      case Aeson.decode contents of
        Just val -> case parseEither parseJSONViaCodec val of
          Right resp -> return $ Just resp
          Left _ -> return Nothing
        Nothing -> return Nothing
    else return Nothing

-- Cached request: check cache first, fall back to real request and cache the result
cachedRequest :: (HasCodec req, HasCodec resp)
              => CachePath
              -> req
              -> IO resp
              -> IO resp
cachedRequest cachePath req makeRequest = do
  cached <- lookupResponse cachePath req
  case cached of
    Just response -> return response
    Nothing -> do
      response <- makeRequest
      recordResponse cachePath req response
      return response

-- Record mode: check cache first, fall back to live API and record response (only caches new responses)
recordMode :: (HasCodec req, HasCodec resp)
           => CachePath
           -> (req -> IO resp)
           -> ResponseProvider req resp
recordMode cachePath apiCall req = cachedRequest cachePath req (apiCall req)

-- Update mode: always make live API call and overwrite cache (updates existing responses)
updateMode :: (HasCodec req, HasCodec resp)
           => CachePath
           -> (req -> IO resp)
           -> ResponseProvider req resp
updateMode cachePath apiCall req = do
  response <- apiCall req
  recordResponse cachePath req response
  return response

-- Playback mode: only use cache, mark test as pending if not found (ensures no unexpected API calls)
-- Cache misses are treated as pending tests rather than failures, since tests may run without cached responses
playbackMode :: (HasCodec req, HasCodec resp)
             => CachePath
             -> ResponseProvider req resp
playbackMode cachePath req = do
  cached <- lookupResponse cachePath req
  case cached of
    Just response -> return response
    Nothing -> do
      pendingWith $ "Cache miss: No cached response for request hash: " ++ hashRequest req
      error "unreachable"  -- pendingWith throws, but GHC doesn't know that

-- Live mode: always make real request, ignore cache
liveMode :: (req -> IO resp) -> ResponseProvider req resp
liveMode apiCall req = apiCall req

-- Cache raw ByteString responses (e.g., SSE streams) without JSON encoding
-- Stores as .sse file instead of .json
-- Also saves the request for debugging purposes
recordRawResponse :: HasCodec req => CachePath -> req -> BSL.ByteString -> IO ()
recordRawResponse cachePath req responseBody = do
  dirExists <- doesDirectoryExist cachePath
  unless dirExists $ error $ "Cache directory does not exist: " ++ cachePath
  let cacheKey = hashRequest req
      responsePath = cachePath </> cacheKey <> ".sse"
      requestPath = cachePath </> cacheKey <> "-request.json"
      requestJson = Aeson.encode $ toJSONViaCodec req
  BSL.writeFile responsePath responseBody
  BSL.writeFile requestPath requestJson

-- Look up a cached raw ByteString response
lookupRawResponse :: HasCodec req => CachePath -> req -> IO (Maybe BSL.ByteString)
lookupRawResponse cachePath req = do
  let cacheKey = hashRequest req
      cachePath' = cachePath </> cacheKey <> ".sse"
  exists <- doesFileExist cachePath'
  if exists
    then Just <$> BSL.readFile cachePath'
    else return Nothing

-- Cached raw request: check cache first, fall back to real request and cache the result
cachedRawRequest :: HasCodec req
                 => CachePath
                 -> req
                 -> IO BSL.ByteString
                 -> IO BSL.ByteString
cachedRawRequest cachePath req makeRequest = do
  cached <- lookupRawResponse cachePath req
  case cached of
    Just response -> return response
    Nothing -> do
      response <- makeRequest
      recordRawResponse cachePath req response
      return response

-- Raw record mode: check cache first, fall back to live API and record response
recordModeRaw :: HasCodec req
              => CachePath
              -> (req -> IO BSL.ByteString)
              -> ResponseProvider req BSL.ByteString
recordModeRaw cachePath apiCall req = cachedRawRequest cachePath req (apiCall req)

-- Raw update mode: always make live API call and overwrite cache
updateModeRaw :: HasCodec req
              => CachePath
              -> (req -> IO BSL.ByteString)
              -> ResponseProvider req BSL.ByteString
updateModeRaw cachePath apiCall req = do
  response <- apiCall req
  recordRawResponse cachePath req response
  return response

-- Raw playback mode: only use cache, mark test as pending if not found
-- Cache misses are treated as pending tests rather than failures, since tests may run without cached responses
playbackModeRaw :: HasCodec req
                => CachePath
                -> ResponseProvider req BSL.ByteString
playbackModeRaw cachePath req = do
  cached <- lookupRawResponse cachePath req
  case cached of
    Just response -> return response
    Nothing -> do
      pendingWith $ "Cache miss: No cached response for request hash: " ++ hashRequest req
      error "unreachable"  -- pendingWith throws, but GHC doesn't know that
