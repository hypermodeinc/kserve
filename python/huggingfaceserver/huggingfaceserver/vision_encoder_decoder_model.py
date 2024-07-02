# Copyright 2024 The KServe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pathlib
from typing import Any, Dict, Optional, Union

import torch
import torch.nn.functional as F
from accelerate import init_empty_weights
from kserve import Model
from kserve.errors import InferenceError
from kserve.logging import logger
from kserve.protocol.infer_type import InferRequest, InferResponse
from kserve.utils.utils import (
    get_predict_input,
    get_predict_response,
)
from torch import Tensor
from transformers import (
    AutoConfig,
    AutoModel,
    AutoProcessor,
    BatchEncoding,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    PretrainedConfig,
    TensorType,
)
from PIL import Image
from io import BytesIO
import base64

from .task import (
    MLTask,
    is_generative_task,
    get_model_class_for_task,
    infer_task_from_model_architecture,
)


class HuggingfaceVisionEncoderDecoderModel(Model):  # pylint:disable=c-extension-no-member
    task: MLTask
    model_config: PretrainedConfig
    model_id_or_path: Union[pathlib.Path, str]
    add_special_tokens: bool
    max_length: Optional[int]
    tensor_input_names: Optional[str]
    return_token_type_ids: Optional[bool]
    model_revision: Optional[str]
    tokenizer_revision: Optional[str]
    trust_remote_code: bool
    ready: bool = False
    _tokenizer: PreTrainedTokenizerBase
    _model: Optional[PreTrainedModel] = None
    _device: torch.device

    def __init__(
        self,
        model_name: str,
        model_id_or_path: Union[pathlib.Path, str],
        model_config: Optional[PretrainedConfig] = None,
        task: Optional[MLTask] = None,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
        tensor_input_names: Optional[str] = None,
        return_token_type_ids: Optional[bool] = None,
        model_revision: Optional[str] = None,
        tokenizer_revision: Optional[str] = None,
        trust_remote_code: bool = False,
    ):
        super().__init__(model_name)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id_or_path = model_id_or_path
        self.add_special_tokens = add_special_tokens
        self.max_length = max_length
        self.dtype = dtype
        self.tensor_input_names = tensor_input_names
        self.return_token_type_ids = return_token_type_ids
        self.model_revision = model_revision
        self.tokenizer_revision = tokenizer_revision
        self.trust_remote_code = trust_remote_code

        if model_config:
            self.model_config = model_config
        else:
            self.model_config = AutoConfig.from_pretrained(self.model_id_or_path)

        if task:
            self.task = task
            try:
                inferred_task = infer_task_from_model_architecture(self.model_config)
            except ValueError:
                inferred_task = None
            if inferred_task is not None and inferred_task != task:
                logger.warn(
                    f"Inferred task is '{inferred_task.name}' but"
                    f" task is explicitly set to '{self.task.name}'"
                )
        else:
            self.task = infer_task_from_model_architecture(self.model_config)

        if is_generative_task(self.task):
            raise RuntimeError(
                f"Encoder model does not support generative task: {self.task.name}"
            )

    def load(self) -> bool:
        model_id_or_path = self.model_id_or_path

        # device_map = "auto" enables model parallelism but all model architcture dont support it.
        # For pre-check we initialize the model class without weights to check the `_no_split_modules`
        # device_map = "auto" for models that support this else set to either cuda/cpu
        with init_empty_weights():
            self._model = AutoModel.from_config(self.model_config)

        device_map = self._device

        if self._model._no_split_modules:
            device_map = "auto"
        # somehow, setting it to True give worse results for NER task
        if self.task == MLTask.token_classification.value:
            self.do_lower_case = False

        processor_kwargs = {}
        model_kwargs = {}

        if self.trust_remote_code:
            model_kwargs["trust_remote_code"] = True
            processor_kwargs["trust_remote_code"] = True

        model_kwargs["torch_dtype"] = self.dtype

        # load huggingface processor
        self._processor = AutoProcessor.from_pretrained(
            str(model_id_or_path),
            revision=self.tokenizer_revision,
            **processor_kwargs,
        )
        logger.info("Successfully loaded processor")

        # load huggingface model using from_pretrained for inference mode
        model_cls = get_model_class_for_task(self.task)
        self._model = model_cls.from_pretrained(
            model_id_or_path,
            revision=self.model_revision,
            device_map=device_map,
            **model_kwargs,
        )
        self._model.eval()
        self._model.to(self._device)
        logger.info(
            f"Successfully loaded huggingface model from path {model_id_or_path}"
        )
        self.ready = True
        return self.ready

    def preprocess(
        self,
        payload: Union[Dict, InferRequest],
        context: Dict[str, Any],
    ) -> Union[BatchEncoding, InferRequest]:
        instances = get_predict_input(payload)
        # for now, can only process one input at a time
        raw_image = Image.open(BytesIO(base64.b64decode(instances[0])))
        inputs = self._processor(
            raw_image,
            return_tensors=TensorType.PYTORCH,
        )
        context["payload"] = payload
        context["inputs"] = inputs
        return inputs

    async def predict(
        self,
        input_batch: Union[BatchEncoding, InferRequest],
        context: Dict[str, Any],
    ) -> Union[Tensor, InferResponse]:
        try:            
            input_batch = input_batch.to(self._device)
            outputs = self._model.generate(**input_batch, max_new_tokens=100) # TODO: max_new_tokens should be configurable
            return outputs
        except Exception as e:
            raise InferenceError(str(e))

    def postprocess(
        self, outputs: Union[Tensor, InferResponse], context: Dict[str, Any]
    ) -> Union[Dict, InferResponse]:
        request = context["payload"]
        if isinstance(outputs, InferResponse):
            shape = torch.Size(outputs.outputs[0].shape)
            data = torch.Tensor(outputs.outputs[0].data)
            outputs = data.view(shape)
        inferences = []
        inferences.append(self._processor.decode(outputs[0], skip_special_tokens=True))
        return get_predict_response(request, inferences, self.name)
