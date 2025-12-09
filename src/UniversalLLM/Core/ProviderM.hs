{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}

module UniversalLLM.Core.ProviderM
  ( ProviderM
  , withProviderState
  , withProvider
  , withProviderCall
  , toRequest
  , fromResponse
  ) where

import Control.Monad.State.Strict (StateT, runStateT, get, put, lift)
import Control.Monad.Trans (MonadTrans)
import Data.Monoid ()
import Data.Default (Default, def)

import UniversalLLM.Core.Types
  ( ComposableProvider
  , toProviderRequest
  , fromProviderResponse
  , ProviderRequest
  , ProviderResponse
  , ModelConfig
  , Message
  )

-- | ProviderM is a monad that threads provider state automatically.
-- It wraps StateT to manage state implicitly while allowing access to
-- arbitrary base monad operations (IO, ExceptT, etc.).
newtype ProviderM model s m a =
  ProviderM (StateT s m a)
  deriving (Functor, Applicative, Monad, MonadTrans)

-- | Core entry point: run the provider monad with explicit state management.
-- Returns both the final state and the result.
withProviderState :: forall model s m a.
                     (Monad m)
                  => ComposableProvider model s
                  -> model
                  -> s
                  -> ProviderM model s m a
                  -> m (s, a)
withProviderState _cp _model initialState (ProviderM stateAction) =
  fmap (\(a, s) -> (s, a)) $ runStateT stateAction initialState

-- | Convenience wrapper: initialize state with Default and discard final state.
withProvider :: forall model s m a.
                (Monad m, Default s)
             => ComposableProvider model s
             -> model
             -> ProviderM model s m a
             -> m a
withProvider cp model action =
  fmap snd $ withProviderState cp model def action

-- | Build a provider request from messages.
-- State is threaded implicitly through the monad.
toRequest :: forall model s m.
             (Monad m, Monoid (ProviderRequest model))
          => ComposableProvider model s
          -> model
          -> [ModelConfig model]
          -> [Message model]
          -> ProviderM model s m (ProviderRequest model)
toRequest composableProvider model configs messages =
  ProviderM $ do
    state <- get
    let (newState, request) = toProviderRequest composableProvider model configs state messages
    put newState
    pure request

-- | Parse a provider response into messages.
-- State is threaded implicitly through the monad.
fromResponse :: forall model s m.
                (MonadFail m)
             => ComposableProvider model s
             -> model
             -> [ModelConfig model]
             -> ProviderResponse model
             -> ProviderM model s m [Message model]
fromResponse composableProvider model configs response =
  ProviderM $ do
    state <- get
    case fromProviderResponse composableProvider model configs state response of
      Left err -> fail $ "LLM error: " ++ show err
      Right (newState, messages) -> do
        put newState
        pure messages

-- | Run with a query function that handles request/response directly.
-- The query function takes messages and returns response messages.
-- Usage: resp <- query [UserText "question"]
--        resp2 <- query (resp <> [UserText "followup"])
withProviderCall :: forall model s m a.
                    (MonadFail m, Default s, Monoid (ProviderRequest model))
                 => ComposableProvider model s
                 -> model
                 -> [ModelConfig model]
                 -> (ProviderRequest model -> m (ProviderResponse model))
                 -> (([Message model] -> ProviderM model s m [Message model]) -> ProviderM model s m a)
                 -> m a
withProviderCall cp model configs callLLM mkProgram =
  withProvider cp model (mkProgram query)
  where
    query :: [Message model] -> ProviderM model s m [Message model]
    query msgs = do
      request <- toRequest cp model configs msgs
      response <- lift $ callLLM request
      fromResponse cp model configs response
