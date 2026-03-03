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
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}


module UniversalLLM.Settings where

import UniversalLLM
import GHC.Generics
import GHC.TypeLits (KnownSymbol, symbolVal)
import Data.Default (Default(..))
import Data.Proxy (Proxy(..))
import Data.Text (Text, pack)


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

-- ============================================================================
-- Runtime Setting Introspection
-- ============================================================================

-- | Runtime representation of any setting value
data SettingValue = BoolValue Bool | DoubleValue Double | IntValue Int
  deriving (Show, Eq)

-- | A single field fully reified for UI display
data SettingField = SettingField
  { sfName        :: Text
  , sfDescription :: Text
  , sfValue       :: SettingValue
  , sfOptional    :: Bool    -- ^ True if the field is Maybe-wrapped
  , sfEnabled     :: Bool    -- ^ For optional fields: True if Just, False if Nothing
  } deriving (Show, Eq)

-- | Bridge setting newtypes to SettingValue
class SettingReify a where
  reifySetting :: a -> SettingValue

instance SettingReify ReasoningSetting where
  reifySetting (ReasoningSetting b) = BoolValue b

instance SettingReify TemperatureSetting where
  reifySetting (TemperatureSetting t) = DoubleValue t

instance SettingReify MaxTokensSetting where
  reifySetting (MaxTokensSetting n) = IntValue n

-- | Inject a SettingValue back into a setting newtype
class SettingInject a where
  injectSetting :: SettingValue -> Maybe a

instance SettingInject ReasoningSetting where
  injectSetting (BoolValue b) = Just (ReasoningSetting b)
  injectSetting _ = Nothing

instance SettingInject TemperatureSetting where
  injectSetting (DoubleValue t) = Just (TemperatureSetting t)
  injectSetting _ = Nothing

instance SettingInject MaxTokensSetting where
  injectSetting (IntValue n) = Just (MaxTokensSetting n)
  injectSetting _ = Nothing

-- ============================================================================
-- Generic Field Introspection
-- ============================================================================

-- | Extract [SettingField] from a generic config record
class GSettingFields f where
  gSettingFields :: f p -> [SettingField]

instance GSettingFields U1 where
  gSettingFields U1 = []

instance (GSettingFields f, GSettingFields g) => GSettingFields (f :*: g) where
  gSettingFields (f :*: g) = gSettingFields f ++ gSettingFields g

instance GSettingFields f => GSettingFields (D1 d f) where
  gSettingFields (M1 x) = gSettingFields x

instance GSettingFields f => GSettingFields (C1 c f) where
  gSettingFields (M1 x) = gSettingFields x

-- Required field: S1 with a non-Maybe type
instance {-# OVERLAPPABLE #-}
  (KnownSymbol name, SettingReify a, SettingDescription a)
  => GSettingFields (S1 ('MetaSel ('Just name) su ss ds) (Rec0 a)) where
  gSettingFields (M1 (K1 x)) =
    [ SettingField
        { sfName = pack (symbolVal (Proxy :: Proxy name))
        , sfDescription = settingDescription @a
        , sfValue = reifySetting x
        , sfOptional = False
        , sfEnabled = True
        }
    ]

-- Optional field: S1 with a Maybe-wrapped type
instance {-# OVERLAPPING #-}
  (KnownSymbol name, SettingReify a, SettingDescription a, Default a)
  => GSettingFields (S1 ('MetaSel ('Just name) su ss ds) (Rec0 (Maybe a))) where
  gSettingFields (M1 (K1 mx)) =
    [ SettingField
        { sfName = pack (symbolVal (Proxy :: Proxy name))
        , sfDescription = settingDescription @a
        , sfValue = case mx of
            Just x  -> reifySetting x
            Nothing -> reifySetting (def :: a)  -- Show default when disabled
        , sfOptional = True
        , sfEnabled = case mx of { Just _ -> True; Nothing -> False }
        }
    ]

-- | Top-level: extract all setting fields from a config record
settingFields :: (Generic cfg, GSettingFields (Rep cfg)) => cfg -> [SettingField]
settingFields cfg = gSettingFields (from cfg)

-- ============================================================================
-- Generic Field Modification
-- ============================================================================

-- | Modify a field by name given a SettingValue
class GSetField f where
  gSetField :: Text -> SettingValue -> f p -> Maybe (f p)

instance GSetField U1 where
  gSetField _ _ _ = Nothing

instance (GSetField f, GSetField g) => GSetField (f :*: g) where
  gSetField name val (f :*: g) =
    case gSetField name val f of
      Just f' -> Just (f' :*: g)
      Nothing -> case gSetField name val g of
        Just g' -> Just (f :*: g')
        Nothing -> Nothing

instance GSetField f => GSetField (D1 d f) where
  gSetField name val (M1 x) = M1 <$> gSetField name val x

instance GSetField f => GSetField (C1 c f) where
  gSetField name val (M1 x) = M1 <$> gSetField name val x

-- Required field
instance {-# OVERLAPPABLE #-}
  (KnownSymbol name, SettingInject a)
  => GSetField (S1 ('MetaSel ('Just name) su ss ds) (Rec0 a)) where
  gSetField fieldName val (M1 (K1 _))
    | fieldName == pack (symbolVal (Proxy :: Proxy name)) =
        case injectSetting val of
          Just a  -> Just (M1 (K1 a))
          Nothing -> Nothing
    | otherwise = Nothing

-- Optional (Maybe) field — set value inside Just
instance {-# OVERLAPPING #-}
  (KnownSymbol name, SettingInject a)
  => GSetField (S1 ('MetaSel ('Just name) su ss ds) (Rec0 (Maybe a))) where
  gSetField fieldName val (M1 (K1 _))
    | fieldName == pack (symbolVal (Proxy :: Proxy name)) =
        case injectSetting val of
          Just a  -> Just (M1 (K1 (Just a)))
          Nothing -> Nothing
    | otherwise = Nothing

-- | Top-level: set a field by name
setField :: (Generic cfg, GSetField (Rep cfg)) => Text -> SettingValue -> cfg -> Maybe cfg
setField name val cfg = to <$> gSetField name val (from cfg)

-- ============================================================================
-- Generic Field Toggle (enable/disable Maybe fields)
-- ============================================================================

-- | Toggle enable/disable for Maybe-wrapped fields
class GToggleField f where
  gToggleField :: Text -> Bool -> f p -> Maybe (f p)

instance GToggleField U1 where
  gToggleField _ _ _ = Nothing

instance (GToggleField f, GToggleField g) => GToggleField (f :*: g) where
  gToggleField name enabled (f :*: g) =
    case gToggleField name enabled f of
      Just f' -> Just (f' :*: g)
      Nothing -> case gToggleField name enabled g of
        Just g' -> Just (f :*: g')
        Nothing -> Nothing

instance GToggleField f => GToggleField (D1 d f) where
  gToggleField name enabled (M1 x) = M1 <$> gToggleField name enabled x

instance GToggleField f => GToggleField (C1 c f) where
  gToggleField name enabled (M1 x) = M1 <$> gToggleField name enabled x

-- Non-Maybe fields can't be toggled
instance {-# OVERLAPPABLE #-}
  GToggleField (S1 ('MetaSel ('Just name) su ss ds) (Rec0 a)) where
  gToggleField _ _ _ = Nothing

-- Maybe fields: enable → Just def, disable → Nothing
instance {-# OVERLAPPING #-}
  (KnownSymbol name, Default a)
  => GToggleField (S1 ('MetaSel ('Just name) su ss ds) (Rec0 (Maybe a))) where
  gToggleField fieldName enabled (M1 (K1 _))
    | fieldName == pack (symbolVal (Proxy :: Proxy name)) =
        Just $ M1 $ K1 $ if enabled then Just def else Nothing
    | otherwise = Nothing

-- | Top-level: toggle an optional field on or off
toggleField :: (Generic cfg, GToggleField (Rep cfg)) => Text -> Bool -> cfg -> Maybe cfg
toggleField name enabled cfg = to <$> gToggleField name enabled (from cfg)