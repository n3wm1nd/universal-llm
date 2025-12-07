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
        let request = defaultOpenAIRequest
              { model = "test-model"
              , messages = [defaultOpenAIMessage
                  { role = "user"
                  , content = Just "test"
                  }]
              }
            response = OpenAISuccess $ defaultOpenAISuccessResponse
              { choices = [defaultOpenAIChoice
                  { message = defaultOpenAIMessage
                      { role = "assistant"
                      , content = Just "test response"
                      }
                  }]
              }

        -- Record the response
        recordResponse cachePath request response

        -- Look it up
        (cached :: Maybe OpenAIResponse) <- lookupResponse cachePath request
        cached `shouldBe` Just response

    it "returns Nothing for cache miss" $
      withSystemTempDirectory "test-cache" $ \cachePath -> do
        let request = defaultOpenAIRequest
              { model = "nonexistent-model"
              , messages = [defaultOpenAIMessage
                  { role = "user"
                  , content = Just "uncached"
                  }]
              }

        (result :: Maybe OpenAIResponse) <- lookupResponse cachePath request
        result `shouldBe` Nothing

    it "errors when cache directory does not exist" $ do
      let cachePath = "/nonexistent/cache/path"
          request = defaultOpenAIRequest { model = "test" }
          response = OpenAISuccess $ defaultOpenAISuccessResponse

      recordResponse cachePath request response `shouldThrow` anyException
