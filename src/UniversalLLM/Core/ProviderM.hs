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
import Data.Monoid (Monoid)
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
newtype ProviderM provider model s m a =
  ProviderM (StateT s m a)
  deriving (Functor, Applicative, Monad, MonadTrans)

-- | Core entry point: run the provider monad with explicit state management.
-- Returns both the final state and the result.
withProviderState :: forall provider model s m a.
                     (Monad m)
                  => ComposableProvider provider model s
                  -> provider
                  -> model
                  -> s
                  -> ProviderM provider model s m a
                  -> m (s, a)
withProviderState _cp _provider _model initialState (ProviderM stateAction) =
  fmap (\(a, s) -> (s, a)) $ runStateT stateAction initialState

-- | Convenience wrapper: initialize state with Default and discard final state.
withProvider :: forall provider model s m a.
                (Monad m, Default s)
             => ComposableProvider provider model s
             -> provider
             -> model
             -> ProviderM provider model s m a
             -> m a
withProvider cp provider model monad =
  fmap snd $ withProviderState cp provider model def monad

-- | Build a provider request from messages.
-- State is threaded implicitly through the monad.
toRequest :: forall provider model s m.
             (Monad m, Monoid (ProviderRequest provider))
          => ComposableProvider provider model s
          -> provider
          -> model
          -> [ModelConfig provider model]
          -> [Message model provider]
          -> ProviderM provider model s m (ProviderRequest provider)
toRequest composableProvider provider model configs messages =
  ProviderM $ do
    state <- get
    let (newState, request) = toProviderRequest composableProvider provider model configs state messages
    put newState
    pure request

-- | Parse a provider response into messages.
-- State is threaded implicitly through the monad.
fromResponse :: forall provider model s m.
                (MonadFail m)
             => ComposableProvider provider model s
             -> provider
             -> model
             -> [ModelConfig provider model]
             -> ProviderResponse provider
             -> ProviderM provider model s m [Message model provider]
fromResponse composableProvider provider model configs response =
  ProviderM $ do
    state <- get
    case fromProviderResponse composableProvider provider model configs state response of
      Left err -> fail $ "LLM error: " ++ show err
      Right (newState, messages) -> do
        put newState
        pure messages

-- | Run with a query function that handles request/response directly.
-- The query function takes messages and returns response messages.
-- Usage: resp <- query [UserText "question"]
--        resp2 <- query (resp <> [UserText "followup"])
withProviderCall :: forall provider model s m a.
                    (MonadFail m, Default s, Monoid (ProviderRequest provider))
                 => ComposableProvider provider model s
                 -> provider
                 -> model
                 -> [ModelConfig provider model]
                 -> (ProviderRequest provider -> m (ProviderResponse provider))
                 -> (([Message model provider] -> ProviderM provider model s m [Message model provider]) -> ProviderM provider model s m a)
                 -> m a
withProviderCall cp provider model configs callLLM mkProgram =
  withProvider cp provider model (mkProgram query)
  where
    query :: [Message model provider] -> ProviderM provider model s m [Message model provider]
    query msgs = do
      request <- toRequest cp provider model configs msgs
      response <- lift $ callLLM request
      fromResponse cp provider model configs response
