"""
测试消融实验模块导入是否正常

运行: python -m src.ablation_studies.test_imports
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

def test_imports():
    """测试所有模块是否能正常导入"""
    
    print("测试消融实验模块导入...\n")
    
    tests = []
    
    # 测试1: 导入配置模块
    try:
        from src.ablation_studies import ablation_config
        print("✅ ablation_config 导入成功")
        print(f"   Device: {ablation_config.BASE_CONFIG['device']}")
        print(f"   Epochs: {ablation_config.BASE_CONFIG['num_epochs']}")
        print(f"   d_model: {ablation_config.BASE_CONFIG['d_model']}")
        tests.append(True)
    except Exception as e:
        print(f"❌ ablation_config 导入失败: {e}")
        tests.append(False)
    
    # 测试2-5: 导入各个消融实验
    experiments = [
        ('ablation_1_num_heads', 'run_ablation_num_heads'),
        ('ablation_2_num_layers', 'run_ablation_num_layers'),
        ('ablation_3_positional_encoding', 'run_ablation_positional_encoding'),
        ('ablation_4_layer_norm', 'run_ablation_layer_norm'),
    ]
    
    for module_name, func_name in experiments:
        try:
            module = __import__(f'src.ablation_studies.{module_name}', fromlist=[func_name])
            func = getattr(module, func_name)
            print(f"✅ {module_name}.{func_name} 导入成功")
            tests.append(True)
        except Exception as e:
            print(f"❌ {module_name}.{func_name} 导入失败: {e}")
            tests.append(False)
    
    # 总结
    print(f"\n{'='*50}")
    success_count = sum(tests)
    total_count = len(tests)
    print(f"测试结果: {success_count}/{total_count} 通过")
    
    if success_count == total_count:
        print("🎉 所有模块导入正常！")
        print("\n可以开始运行消融实验了:")
        print("  python -m src.ablation_studies.ablation_1_num_heads")
        print("  python -m src.ablation_studies.ablation_2_num_layers")
        print("  python -m src.ablation_studies.ablation_3_positional_encoding")
        print("  python -m src.ablation_studies.ablation_4_layer_norm")
    else:
        print("⚠️  部分模块导入失败，请检查错误信息")
    
    print(f"{'='*50}\n")
    
    return success_count == total_count


if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
