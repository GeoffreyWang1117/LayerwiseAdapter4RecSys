#!/usr/bin/env python3
"""
项目概览脚本 - Amazon LLM推荐系统
快速查看项目结构、实验结果和模型性能
"""

import json
from pathlib import Path

def print_header(title):
    """打印标题"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def print_section(title):
    """打印章节"""
    print(f"\n📋 {title}")
    print("-" * 40)

def analyze_project_structure():
    """分析项目结构"""
    print_header("Amazon LLM推荐系统 - 项目概览")
    
    base_dir = Path(__file__).parent
    
    print_section("项目结构")
    
    # 统计文件数量
    src_files = list((base_dir / "src").glob("*.py")) if (base_dir / "src").exists() else []
    doc_files = list((base_dir / "docs").glob("*.md")) if (base_dir / "docs").exists() else []
    result_files = list((base_dir / "results").glob("*.json")) if (base_dir / "results").exists() else []
    
    print(f"📁 源代码文件: {len(src_files)} 个")
    for f in src_files:
        print(f"   • {f.name}")
    
    print(f"\n📄 文档文件: {len(doc_files)} 个")
    for f in doc_files:
        print(f"   • {f.name}")
    
    print(f"\n📊 结果文件: {len(result_files)} 个")
    for f in result_files:
        print(f"   • {f.name}")
    
    return base_dir, result_files

def analyze_experiment_results(result_files):
    """分析实验结果"""
    print_section("实验结果摘要")
    
    if not result_files:
        print("❌ 未找到实验结果文件")
        return
    
    for result_file in result_files:
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"\n🔬 {result_file.name}")
            
            if 'multi_model_comparison' in result_file.name:
                # 多模型对比结果
                models = data.get('experiment_info', {}).get('models_available', {})
                categories = list(data.get('category_results', {}).keys())
                
                print(f"   🤖 测试模型: {len(models)} 个 ({', '.join(models.keys())})")
                print(f"   📦 测试类别: {len(categories)} 个 ({', '.join(categories)})")
                
                if 'summary_statistics' in data:
                    summary = data['summary_statistics'].get('model_performance', {})
                    print("   ⚡ 性能最优: ", end="")
                    if summary:
                        best_model = min(summary.items(), key=lambda x: x[1].get('avg_response_time', float('inf')))
                        print(f"{best_model[0]} ({best_model[1].get('avg_response_time', 0):.2f}s)")
                    else:
                        print("数据解析中...")
                        
            elif 'multi_category' in result_file.name:
                # 多类别推荐结果
                if isinstance(data, dict):
                    recommendations = data.get('recommendations', {})
                    print(f"   👥 推荐用户: {len(recommendations)} 个")
                    if recommendations:
                        categories = set()
                        for user_data in recommendations.values():
                            categories.add(user_data.get('category', 'Unknown'))
                        print(f"   📦 涉及类别: {', '.join(categories)}")
                else:
                    print(f"   📊 数据条目: {len(data)} 个")
                    
            else:
                # 单一推荐结果
                if isinstance(data, dict):
                    if 'user_analysis' in data:
                        category = data.get('category', 'Unknown')
                        user_count = len(data.get('recommendations', []))
                        print(f"   📦 类别: {category}")
                        print(f"   🎯 推荐数量: {user_count} 个")
                    else:
                        print(f"   📊 包含字段: {', '.join(data.keys())}")
                else:
                    print(f"   📊 数据条目: {len(data)} 个")
                    
        except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
            print(f"   ❌ 解析失败: {e}")

def check_model_availability():
    """检查ollama模型可用性"""
    print_section("模型可用性检查")
    
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [model["name"] for model in models]
            
            target_models = ["qwen3:latest", "llama3:latest", "gpt-oss:latest"]
            
            print("🤖 Ollama服务状态: ✅ 运行中")
            print(f"📊 可用模型总数: {len(models)}")
            
            print("\n🎯 项目所需模型:")
            for model in target_models:
                if model in model_names:
                    print(f"   ✅ {model}")
                else:
                    print(f"   ❌ {model} (未安装)")
                    
        else:
            print("❌ Ollama服务无响应")
            
    except (requests.RequestException, ConnectionError, TimeoutError) as e:
        print(f"❌ 无法连接到Ollama服务: {e}")
        print("💡 请确保ollama已启动: ollama serve")

def show_quick_commands():
    """显示快速命令"""
    print_section("快速使用命令")
    
    print("🚀 运行推荐系统:")
    print("   cd src")
    print("   python multi_category_recommender.py     # 多类别推荐(推荐)")
    print("   python enhanced_amazon_recommender.py    # 单类别推荐")
    print("   python multi_model_comparison.py         # 模型对比实验")
    
    print("\n📚 查看文档:")
    print("   cat docs/PROJECT_FINAL_SUMMARY.md        # 项目总结")
    print("   cat docs/MULTI_MODEL_COMPARISON_REPORT.md # 模型对比报告")
    print("   cat docs/EXPERIMENT_REPORT.md            # 实验报告")
    
    print("\n📊 查看结果:")
    print("   ls -la results/                          # 所有实验结果")
    print("   cat results/multi_model_comparison_*.json # 模型对比数据")

def main():
    """主函数"""
    try:
        _, result_files = analyze_project_structure()
        analyze_experiment_results(result_files)
        check_model_availability()
        show_quick_commands()
        
        print_header("项目概览完成")
        print("🎉 Amazon LLM推荐系统已就绪!")
        print("📖 详细信息请查看 README.md")
        
    except (OSError, FileNotFoundError) as e:
        print(f"❌ 概览生成失败: {e}")

if __name__ == "__main__":
    main()
