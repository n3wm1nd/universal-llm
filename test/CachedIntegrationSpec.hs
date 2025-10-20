{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}

module CachedIntegrationSpec (spec) where

import Test.Hspec
import TestCache
import UniversalLLM.Protocols.OpenAI
import qualified Data.Aeson as Aeson
import System.IO.Temp (withSystemTempDirectory)

spec :: Spec
spec = do
  describe "Test Cache" $ do
    it "stores and retrieves responses" $
      withSystemTempDirectory "test-cache" $ \cachePath -> do
        let request = OpenAIRequest
              { model = "test-model"
              , messages = [OpenAIMessage "user" (Just "test") Nothing Nothing]
              , temperature = Nothing
              , max_tokens = Nothing
              , seed = Nothing
              , tools = Nothing
              , response_format = Nothing
              }
            response = OpenAISuccess $ OpenAISuccessResponse
              [ OpenAIChoice $ OpenAIMessage "assistant" (Just "test response") Nothing Nothing ]

        -- Record the response
        recordResponse cachePath request response

        -- Look it up
        (cached :: Maybe OpenAIResponse) <- lookupResponse cachePath request
        cached `shouldBe` Just response

    it "returns Nothing for cache miss" $
      withSystemTempDirectory "test-cache" $ \cachePath -> do
        let request = OpenAIRequest
              { model = "nonexistent-model"
              , messages = [OpenAIMessage "user" (Just "uncached") Nothing Nothing]
              , temperature = Nothing
              , max_tokens = Nothing
              , seed = Nothing
              , tools = Nothing
              , response_format = Nothing
              }

        (result :: Maybe OpenAIResponse) <- lookupResponse cachePath request
        result `shouldBe` Nothing

    it "errors when cache directory does not exist" $ do
      let cachePath = "/nonexistent/cache/path"
          request = OpenAIRequest
            { model = "test"
            , messages = []
            , temperature = Nothing
            , max_tokens = Nothing
            , seed = Nothing
            , tools = Nothing
            , response_format = Nothing
            }
          response = OpenAISuccess $ OpenAISuccessResponse []

      recordResponse cachePath request response `shouldThrow` anyException
