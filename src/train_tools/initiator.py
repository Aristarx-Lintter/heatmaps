import torch.nn as nn

from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2RMSNorm


def init_transformer_block_weights(block, std=0.02):
    """
    Применяет инициализацию к Linear, LayerNorm и RMSNorm слоям в блоке.
    """
    print(f"--- Initializing block: {block.__class__.__name__} with std={std} ---")
    initialized_modules = 0
    skipped_types = set()

    for name, module in block.named_modules(): # Используем named_modules для лучшего логгирования
        module_type_name = module.__class__.__name__

        if isinstance(module, nn.Linear):
            print(f"  Initializing Linear weights: {name} ({module_type_name})")
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                print(f"  Initializing Linear bias: {name} ({module_type_name})")
                module.bias.data.zero_()
            initialized_modules += 1
        elif isinstance(module, nn.LayerNorm):
            print(f"  Initializing LayerNorm: {name} ({module_type_name})")
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            initialized_modules += 1
        # ----- НОВОЕ: Обработка RMSNorm -----
        elif Qwen2RMSNorm is not None and isinstance(module, Qwen2RMSNorm):
            print(f"  Initializing RMSNorm weight: {name} ({module_type_name})")
            if hasattr(module, 'weight') and module.weight is not None:
                 module.weight.data.fill_(1.0)
            else:
                 print(f"    WARNING: RMSNorm module {name} does not have 'weight' attribute or it's None.")
            # RMSNorm обычно не имеет bias, поэтому его не инициализируем
            initialized_modules += 1
        # ------------------------------------
        elif name == "": # Пропускаем сам корневой блок
             pass
        else:
             # Логируем типы модулей, которые НЕ были инициализированы (кроме базовых контейнеров)
             is_container = isinstance(module, (nn.ModuleList, nn.Sequential, nn.ModuleDict)) or len(list(module.children())) > 0
             if not is_container and module_type_name not in skipped_types:
                  # print(f"  Skipping module type: {module_type_name} (name: {name})")
                  skipped_types.add(module_type_name)


    print(f"--- Finished initializing block. Processed {initialized_modules} Linear/LayerNorm/RMSNorm modules. ---")
    if skipped_types:
        print(f"--- Skipped module types encountered: {', '.join(skipped_types)} ---")