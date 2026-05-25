{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RankNTypes #-}

module TestCache
  ( CachePath
  , ResponseProvider
  , IsExpected
  , IsTransient
  , request
  , recordResponse
  , lookupResponse
  , recordMode
  , recordModeWithFilterMsg
  , updateMode
  , updateModeWithFilterMsg
  , playbackMode
  , liveMode
  , liveModeWithFilterMsg
  , recordRawResponse
  , lookupRawResponse
  , recordModeRaw
  , recordModeRawWithFilterMsg
  , updateModeRaw
  , updateModeRawWithFilterMsg
  , playbackModeRaw
  , CacheMissException(..)
  ) where

import qualified Data.ByteString.Lazy as BSL
import qualified Data.ByteString.Char8 as BS8
import qualified Crypto.Hash.SHA256 as SHA256
import qualified Data.ByteString.Base16 as Base16
import System.Directory (doesDirectoryExist, doesFileExist)
import System.FilePath ((</>))
import Control.Monad (unless)
import Control.Exception (Exception)
import Autodocodec (HasCodec, toJSONViaCodec, parseJSONViaCodec)
import Data.Aeson.Types (parseEither)
import qualified Data.Aeson as Aeson
import Test.Hspec (pendingWith)

data CacheMissException = CacheMissException String
  deriving (Show)

instance Exception CacheMissException

type CachePath = FilePath

-- | Per-request predicate: "is this the response I expected?"
-- When True, always cache. When False and also transient, skip as pending.
type IsExpected resp = Int -> resp -> Bool

-- | Per-provider predicate: "is this a transient server-side error?"
-- Transient responses are not cached unless the probe explicitly expected them.
type IsTransient resp = Int -> resp -> Bool

-- | A provider is a function from (expectation predicate, request) to response.
-- Each call site decides what it expects — the provider never does.
type ResponseProvider req resp = IsExpected resp -> req -> IO resp

-- | Make a request expecting HTTP 200. Sugar for the common case.
request :: ResponseProvider req resp -> req -> IO resp
request provider req = provider (\sc _ -> sc == 200) req

hashRequest :: HasCodec req => String -> req -> String
hashRequest endpoint req =
  let combined = BS8.pack endpoint <> BSL.toStrict (Aeson.encode $ toJSONViaCodec req)
  in BS8.unpack $ Base16.encode $ SHA256.hash combined

recordResponse :: (HasCodec req, HasCodec resp) => CachePath -> String -> req -> resp -> IO ()
recordResponse cachePath endpoint req resp = do
  dirExists <- doesDirectoryExist cachePath
  unless dirExists $ error $ "Cache directory does not exist: " ++ cachePath
  let key = hashRequest endpoint req
  BSL.writeFile (cachePath </> key <> ".json")             (Aeson.encode $ toJSONViaCodec resp)
  BSL.writeFile (cachePath </> key <> "-request.json") (Aeson.encode $ toJSONViaCodec req)

lookupResponse :: (HasCodec req, HasCodec resp) => CachePath -> String -> req -> IO (Maybe resp)
lookupResponse cachePath endpoint req = do
  let path = cachePath </> hashRequest endpoint req <> ".json"
  exists <- doesFileExist path
  if not exists then return Nothing else do
    contents <- BSL.readFile path
    case Aeson.decode contents of
      Just val -> case parseEither parseJSONViaCodec val of
        Right resp -> return $ Just resp
        Left _     -> return Nothing
      Nothing -> return Nothing

handleResponse :: (HasCodec req, HasCodec resp)
               => CachePath -> String -> IsTransient resp -> IsExpected resp
               -> req -> Int -> resp -> IO resp
handleResponse cachePath endpoint isTransient isExpected req statusCode response =
  if isExpected statusCode response || not (isTransient statusCode response)
    then recordResponse cachePath endpoint req response >> return response
    else do
      pendingWith $ "Transient error (status " ++ show statusCode ++ ") — not caching"
      error "unreachable"

cachedCall :: (HasCodec req, HasCodec resp)
           => CachePath -> String -> IsTransient resp -> IsExpected resp
           -> req -> IO (Int, resp) -> IO resp
cachedCall cachePath endpoint isTransient isExpected req makeRequest = do
  cached <- lookupResponse cachePath endpoint req
  case cached of
    Just response -> return response
    Nothing -> do
      (statusCode, response) <- makeRequest
      handleResponse cachePath endpoint isTransient isExpected req statusCode response

recordMode :: (HasCodec req, HasCodec resp)
           => CachePath -> String -> IsTransient resp -> (req -> IO (Int, resp))
           -> ResponseProvider req resp
recordMode cachePath endpoint isTransient apiCall isExpected req =
  cachedCall cachePath endpoint isTransient isExpected req (apiCall req)

updateMode :: (HasCodec req, HasCodec resp)
           => CachePath -> String -> IsTransient resp -> (req -> IO (Int, resp))
           -> ResponseProvider req resp
updateMode cachePath endpoint isTransient apiCall isExpected req = do
  (statusCode, response) <- apiCall req
  handleResponse cachePath endpoint isTransient isExpected req statusCode response

playbackMode :: (HasCodec req, HasCodec resp)
             => CachePath -> String
             -> ResponseProvider req resp
playbackMode cachePath endpoint _isExpected req = do
  cached <- lookupResponse cachePath endpoint req
  case cached of
    Just response -> return response
    Nothing -> do
      pendingWith $ "Cache miss: No cached response for request hash: " ++ hashRequest endpoint req
      error "unreachable"

liveMode :: (req -> IO (Int, resp)) -> ResponseProvider req resp
liveMode apiCall _isExpected req = snd <$> apiCall req

recordModeWithFilterMsg :: (HasCodec req, HasCodec resp)
                        => CachePath -> String -> IsTransient resp
                        -> (req -> (Bool, String)) -> (req -> IO (Int, resp))
                        -> ResponseProvider req resp
recordModeWithFilterMsg cachePath endpoint isTransient filterFn apiCall isExpected req = do
  cached <- lookupResponse cachePath endpoint req
  case cached of
    Just response -> return response
    Nothing -> case filterFn req of
      (True, _)    -> cachedCall cachePath endpoint isTransient isExpected req (apiCall req)
      (False, msg) -> pendingWith msg >> error "unreachable"

updateModeWithFilterMsg :: (HasCodec req, HasCodec resp)
                        => CachePath -> String -> IsTransient resp
                        -> (req -> (Bool, String)) -> (req -> IO (Int, resp))
                        -> ResponseProvider req resp
updateModeWithFilterMsg cachePath endpoint isTransient filterFn apiCall isExpected req =
  case filterFn req of
    (True, _)    -> do
      (statusCode, response) <- apiCall req
      handleResponse cachePath endpoint isTransient isExpected req statusCode response
    (False, msg) -> pendingWith msg >> error "unreachable"

liveModeWithFilterMsg :: (req -> (Bool, String)) -> (req -> IO (Int, resp))
                      -> ResponseProvider req resp
liveModeWithFilterMsg filterFn apiCall _isExpected req =
  case filterFn req of
    (True, _)    -> snd <$> apiCall req
    (False, msg) -> pendingWith msg >> error "unreachable"

-- Raw variants (SSE/ByteString)

recordRawResponse :: HasCodec req => CachePath -> String -> req -> BSL.ByteString -> IO ()
recordRawResponse cachePath endpoint req responseBody = do
  dirExists <- doesDirectoryExist cachePath
  unless dirExists $ error $ "Cache directory does not exist: " ++ cachePath
  let key = hashRequest endpoint req
  BSL.writeFile (cachePath </> key <> ".sse")              responseBody
  BSL.writeFile (cachePath </> key <> "-request.json") (Aeson.encode $ toJSONViaCodec req)

lookupRawResponse :: HasCodec req => CachePath -> String -> req -> IO (Maybe BSL.ByteString)
lookupRawResponse cachePath endpoint req = do
  let path = cachePath </> hashRequest endpoint req <> ".sse"
  exists <- doesFileExist path
  if exists then Just <$> BSL.readFile path else return Nothing

handleRawResponse :: HasCodec req
                  => CachePath -> String -> IsTransient BSL.ByteString -> IsExpected BSL.ByteString
                  -> req -> Int -> BSL.ByteString -> IO BSL.ByteString
handleRawResponse cachePath endpoint isTransient isExpected req statusCode response =
  if isExpected statusCode response || not (isTransient statusCode response)
    then recordRawResponse cachePath endpoint req response >> return response
    else do
      pendingWith $ "Transient error (status " ++ show statusCode ++ ") — not caching"
      error "unreachable"

recordModeRaw :: HasCodec req
              => CachePath -> String -> IsTransient BSL.ByteString -> (req -> IO (Int, BSL.ByteString))
              -> ResponseProvider req BSL.ByteString
recordModeRaw cachePath endpoint isTransient apiCall isExpected req = do
  cached <- lookupRawResponse cachePath endpoint req
  case cached of
    Just response -> return response
    Nothing -> do
      (statusCode, response) <- apiCall req
      handleRawResponse cachePath endpoint isTransient isExpected req statusCode response

updateModeRaw :: HasCodec req
              => CachePath -> String -> IsTransient BSL.ByteString -> (req -> IO (Int, BSL.ByteString))
              -> ResponseProvider req BSL.ByteString
updateModeRaw cachePath endpoint isTransient apiCall isExpected req = do
  (statusCode, response) <- apiCall req
  handleRawResponse cachePath endpoint isTransient isExpected req statusCode response

playbackModeRaw :: HasCodec req
                => CachePath -> String
                -> ResponseProvider req BSL.ByteString
playbackModeRaw cachePath endpoint _isExpected req = do
  cached <- lookupRawResponse cachePath endpoint req
  case cached of
    Just response -> return response
    Nothing -> do
      pendingWith $ "Cache miss: No cached response for request hash: " ++ hashRequest endpoint req
      error "unreachable"

recordModeRawWithFilterMsg :: HasCodec req
                           => CachePath -> String -> IsTransient BSL.ByteString
                           -> (req -> (Bool, String)) -> (req -> IO (Int, BSL.ByteString))
                           -> ResponseProvider req BSL.ByteString
recordModeRawWithFilterMsg cachePath endpoint isTransient filterFn apiCall isExpected req = do
  cached <- lookupRawResponse cachePath endpoint req
  case cached of
    Just response -> return response
    Nothing -> case filterFn req of
      (True, _)    -> do
        (statusCode, response) <- apiCall req
        handleRawResponse cachePath endpoint isTransient isExpected req statusCode response
      (False, msg) -> pendingWith msg >> error "unreachable"

updateModeRawWithFilterMsg :: HasCodec req
                           => CachePath -> String -> IsTransient BSL.ByteString
                           -> (req -> (Bool, String)) -> (req -> IO (Int, BSL.ByteString))
                           -> ResponseProvider req BSL.ByteString
updateModeRawWithFilterMsg cachePath endpoint isTransient filterFn apiCall isExpected req =
  case filterFn req of
    (True, _)    -> do
      (statusCode, response) <- apiCall req
      handleRawResponse cachePath endpoint isTransient isExpected req statusCode response
    (False, msg) -> pendingWith msg >> error "unreachable"
