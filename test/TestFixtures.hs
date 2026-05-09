{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TupleSections #-}

module TestFixtures
  ( loadImageBase64
  , glassbottlePng
  , glassbottleMirroredJpeg
  ) where

import Data.Text (Text)
import qualified Data.Text.Encoding as TE
import qualified Data.ByteString as BS
import qualified Data.ByteString.Base64 as B64
import Paths_universal_llm (getDataFileName)

-- | Load a fixture image and return it as base64-encoded text.
-- Path is relative to the package data-files root (test/fixtures/).
loadImageBase64 :: FilePath -> IO Text
loadImageBase64 relPath = do
  absPath <- getDataFileName ("test/fixtures/" <> relPath)
  bytes <- BS.readFile absPath
  return $ TE.decodeUtf8 (B64.encode bytes)

-- | Load glassbottle.png as (mediaType, base64Data)
glassbottlePng :: IO (Text, Text)
glassbottlePng = ("image/png",) <$> loadImageBase64 "glassbottle.png"

-- | Load glassbottle-mirrored.jpg as (mediaType, base64Data)
glassbottleMirroredJpeg :: IO (Text, Text)
glassbottleMirroredJpeg = ("image/jpeg",) <$> loadImageBase64 "glassbottle-mirrored.jpg"
