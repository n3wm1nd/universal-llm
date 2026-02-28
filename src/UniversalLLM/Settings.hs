{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE OverloadedStrings #-}


module UniversalLLM.Settings where

import UniversalLLM
import GHC.Generics
import Data.Default (Default(..))
import Data.Text


-- ============================================================================
-- Generic Configuration System
-- ============================================================================

-- | Setting types - newtypes wrapping configuration values
-- These can be used to build type-safe config records for models
-- NOTE: Streaming is NOT a setting - it's determined by which interpreter is used
-- (interpretLLM for non-streaming, interpretLLMStream for streaming)

newtype ReasoningSetting = ReasoningSetting Bool
  deriving (Show, Eq, Generic)

newtype TemperatureSetting = TemperatureSetting Double
  deriving (Show, Eq, Generic)

newtype MaxTokensSetting = MaxTokensSetting Int
  deriving (Show, Eq, Generic)

-- | Setting metadata - descriptions for UI/CLI rendering
class SettingDescription setting where
  settingDescription :: Text

instance SettingDescription ReasoningSetting where
  settingDescription = "Enable reasoning extraction"

instance SettingDescription TemperatureSetting where
  settingDescription = "Sampling temperature (0.0-1.0)"

instance SettingDescription MaxTokensSetting where
  settingDescription = "Maximum tokens to generate"

-- | Default values for settings
instance Default ReasoningSetting where
  def = ReasoningSetting True

instance Default TemperatureSetting where
  def = TemperatureSetting 0.7

instance Default MaxTokensSetting where
  def = MaxTokensSetting 4096

-- | Apply a setting to generate a ModelConfig value
-- Instances are constraint-based, ensuring only valid settings can be applied
class ApplySetting setting model where
  applySetting :: setting -> ModelConfig model

instance HasReasoning model => ApplySetting ReasoningSetting model where
  applySetting (ReasoningSetting b) = Reasoning b

instance SupportsTemperature (ProviderOf model) => ApplySetting TemperatureSetting model where
  applySetting (TemperatureSetting t) = Temperature t

instance SupportsMaxTokens (ProviderOf model) => ApplySetting MaxTokensSetting model where
  applySetting (MaxTokensSetting n) = MaxTokens n

-- | Generic machinery for applying all settings in a config record
class GApplySettings f model where
  gApplySettings :: f p -> [ModelConfig model]

instance GApplySettings U1 model where
  gApplySettings U1 = []

instance (GApplySettings f model, GApplySettings g model)
  => GApplySettings (f :*: g) model where
  gApplySettings (f :*: g) = gApplySettings f ++ gApplySettings g

instance GApplySettingsField a model => GApplySettings (S1 s (Rec0 a)) model where
  gApplySettings (M1 (K1 x)) = gApplySettingsField x

instance GApplySettings f model => GApplySettings (D1 d f) model where
  gApplySettings (M1 x) = gApplySettings x

instance GApplySettings f model => GApplySettings (C1 c f) model where
  gApplySettings (M1 x) = gApplySettings x

-- | Helper to handle Maybe vs non-Maybe fields
class GApplySettingsField a model where
  gApplySettingsField :: a -> [ModelConfig model]

instance {-# OVERLAPPABLE #-} ApplySetting a model => GApplySettingsField a model where
  gApplySettingsField x = [applySetting x]

instance {-# OVERLAPPING #-} ApplySetting a model => GApplySettingsField (Maybe a) model where
  gApplySettingsField Nothing = []
  gApplySettingsField (Just x) = [applySetting x]

-- | Generic default config derivation
class GDefault f where
  gDefault :: f p

instance GDefault U1 where
  gDefault = U1

instance (GDefault f, GDefault g) => GDefault (f :*: g) where
  gDefault = gDefault :*: gDefault

instance Default a => GDefault (S1 s (Rec0 a)) where
  gDefault = M1 (K1 def)

instance GDefault f => GDefault (D1 d f) where
  gDefault = M1 gDefault

instance GDefault f => GDefault (C1 c f) where
  gDefault = M1 gDefault

-- | Get default config for any config record with Default instances for all fields
defaultConfig :: (Generic cfg, GDefault (Rep cfg)) => cfg
defaultConfig = to gDefault

-- | Constraint for valid model configuration records
-- A config record must be Generic and all fields must be ApplySetting instances
type ModelSettings cfg model = (Generic cfg, GApplySettings (Rep cfg) model)

-- | Convert a config record to [ModelConfig model] using Generic derivation
toModelConfigs :: ModelSettings cfg model => cfg -> [ModelConfig model]
toModelConfigs cfg = gApplySettings (from cfg)

-- ============================================================================
-- ConfigFor Type Family
-- ============================================================================

-- | Open type family mapping models to their canonical configuration types
-- Each model can define its own configuration record type and register it here.
-- The canonical config is expected to always work for the model, though other
-- compatible config types may also be used.
type family ConfigFor model