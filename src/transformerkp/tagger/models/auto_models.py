from transformers import (
    AutoModelForTokenClassification,
    PretrainedConfig,
    AutoConfig,
)
from .models import BertCrfModelForKpExtraction, RobertaCrfForKpExtraction

CRF_MODEL_MAPPINGS = {
    "BertConfig": BertCrfModelForKpExtraction,
    "RobertaConfig": RobertaCrfForKpExtraction,
}


class AutoModelForKPExtraction(AutoModelForTokenClassification):
    pass


class AutoCrfModelForKPExtraction:
    def __init__(self, *args, **kwargs) -> None:
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated "
            f"using the `{self.__class__.__name__}.from_pretrained(model_name_or_path)` or "
            f"`{self.__class__.__name__}.from_config(config)` methods."
        )

    @classmethod
    def from_config(cls, config, **kwargs):
        if config.__class__.__name__ in CRF_MODEL_MAPPINGS.keys():
            model_class = CRF_MODEL_MAPPINGS[config.__class__.__name__]
            return model_class._from_config(config, **kwargs)

        raise ValueError(
            f"Unrecognized configuration class {config.__class__} for this kind of AutoModel: {cls.__name__}.\n"
            f"Model type should be one of {', '.join(c for c in CRF_MODEL_MAPPINGS.keys())}."
        )

    @classmethod
    def from_pretrained(cls, model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)
        trust_remote_code = kwargs.pop("trust_remote_code", False)
        if not isinstance(config, PretrainedConfig):
            config, kwargs = AutoConfig.from_pretrained(
                model_name_or_path,
                return_unused_kwargs=True,
                **kwargs,
            )
        if config.__class__.__name__ in CRF_MODEL_MAPPINGS.keys():
            model_class = CRF_MODEL_MAPPINGS[config.__class__.__name__]
            return model_class.from_pretrained(
                model_name_or_path, *model_args, config=config, **kwargs
            )
        raise ValueError(
            f"Unrecognized configuration class {config.__class__} for this kind of AutoModel: {cls.__name__}.\n"
            f"Model type should be one of {', '.join(c for c in CRF_MODEL_MAPPINGS.keys())}."
        )
