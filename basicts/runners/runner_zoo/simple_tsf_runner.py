from typing import Dict, Optional

import torch

from ..base_tsf_runner import BaseTimeSeriesForecastingRunner
from case_heatmap_utils import export_case_heatmaps_from_current_batch


class SimpleTimeSeriesForecastingRunner(BaseTimeSeriesForecastingRunner):
    """
    A Simple Runner for Time Series Forecasting.
    """

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        self.forward_features = cfg['MODEL'].get('FORWARD_FEATURES', None)
        self.target_features = cfg['MODEL'].get('TARGET_FEATURES', None)
        self.target_time_series = cfg['MODEL'].get('TARGET_TIME_SERIES', None)

        case_cfg = cfg.get('CASE_HEATMAP', {})
        self.case_heatmap_enabled = bool(case_cfg.get('ENABLED', False))
        self.case_heatmap_output_dir = case_cfg.get('OUTPUT_DIR', 'case_results')
        self.case_heatmap_sample_index = case_cfg.get('SAMPLE_INDEX', None)
        self.case_heatmap_top_k = case_cfg.get('TOP_K', None)
        self.case_heatmap_static_binary = bool(case_cfg.get('STATIC_BINARY', True))
        self.case_heatmap_use_raw_lcdg = bool(case_cfg.get('USE_RAW_LCDG', True))
        self.case_heatmap_dpi = int(case_cfg.get('DPI', 600))
        self._case_heatmap_exported = False

    def preprocessing(self, input_data: Dict) -> Dict:
        if self.scaler is not None:
            input_data['target'] = self.scaler.transform(input_data['target'])
            input_data['inputs'] = self.scaler.transform(input_data['inputs'])
        return input_data

    def postprocessing(self, input_data: Dict) -> Dict:
        if self.scaler is not None and self.scaler.rescale:
            input_data['prediction'] = self.scaler.inverse_transform(input_data['prediction'])
            input_data['target'] = self.scaler.inverse_transform(input_data['target'])
            input_data['inputs'] = self.scaler.inverse_transform(input_data['inputs'])

        if self.target_time_series is not None:
            input_data['target'] = input_data['target'][:, :, self.target_time_series, :]
            input_data['prediction'] = input_data['prediction'][:, :, self.target_time_series, :]
        return input_data

    def forward(
        self,
        data: Dict,
        epoch: Optional[int] = None,
        iter_num: Optional[int] = None,
        train: bool = True,
        **kwargs
    ) -> Dict:
        data = self.preprocessing(data)

        future_data, history_data = data['target'], data['inputs']
        history_data = self.to_running_device(history_data)
        future_data = self.to_running_device(future_data)
        batch_size, length, num_nodes, _ = future_data.shape

        history_data = self.select_input_features(history_data)
        future_data_4_dec = self.select_input_features(future_data)

        if not train:
            future_data_4_dec[..., 0] = torch.empty_like(future_data_4_dec[..., 0])

        model_return = self.model(
            history_data=history_data,
            future_data=future_data_4_dec,
            batch_seen=iter_num,
            epoch=epoch,
            train=train
        )

        if (
            self.case_heatmap_enabled
            and not train
            and not self._case_heatmap_exported
        ):
            chosen_index = export_case_heatmaps_from_current_batch(
                model=self.model,
                history_data=history_data,
                output_dir=self.case_heatmap_output_dir,
                sample_index=self.case_heatmap_sample_index,
                top_k=self.case_heatmap_top_k,
                static_binary=self.case_heatmap_static_binary,
                use_raw_lcdg=self.case_heatmap_use_raw_lcdg,
                dpi=self.case_heatmap_dpi,
            )
            self._case_heatmap_exported = True
            print(
                "\n[Case Heatmap] 三种图已经导出："
                f"{self.case_heatmap_output_dir}\n"
                f"[Case Heatmap] 当前 batch 选中的样本下标：{chosen_index}\n"
            )

        if isinstance(model_return, torch.Tensor):
            model_return = {'prediction': model_return}
        if 'inputs' not in model_return:
            model_return['inputs'] = self.select_target_features(history_data)
        if 'target' not in model_return:
            model_return['target'] = self.select_target_features(future_data)

        assert list(model_return['prediction'].shape)[:3] == [
            batch_size, length, num_nodes
        ], "The shape of the output is incorrect. Ensure it matches [B, L, N, C]."

        model_return = self.postprocessing(model_return)
        return model_return

    def select_input_features(self, data: torch.Tensor) -> torch.Tensor:
        if self.forward_features is not None:
            data = data[:, :, :, self.forward_features]
        return data

    def select_target_features(self, data: torch.Tensor) -> torch.Tensor:
        data = data[:, :, :, self.target_features]
        return data

    def select_target_time_series(self, data: torch.Tensor) -> torch.Tensor:
        data = data[:, :, self.target_time_series, :]
        return data