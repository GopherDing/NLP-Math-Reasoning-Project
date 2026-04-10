"""
Streamlit Web UI for Math Reasoning Experiments
"""

import streamlit as st
import sys
import os
import json
from pathlib import Path

# Configure HuggingFace mirror for users behind the Great Firewall
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from models.loader import load_model
from prompts.cot import solve_cot
from prompts.self_refine import solve_self_refine
from prompts.self_consistency import solve_self_consistency

st.set_page_config(page_title="Math Reasoning Demo", layout="wide")

# Load all dataset examples globally (shared across pages)
@st.cache_data(ttl=3600)
def load_dataset_examples():
    examples = {"自定义输入": "", "GSM8K": [], "MATH-500": [], "AIME-2024": []}
    warnings = []

    try:
        from data.loader import load_dataset_by_name
        gsm8k = load_dataset_by_name("gsm8k")
        examples["GSM8K"] = [(item["problem"], item["answer"]) for item in gsm8k[:5]]
    except Exception as e:
        warnings.append(f"GSM8K: {e}")

    try:
        from data.loader import load_dataset_by_name
        math500 = load_dataset_by_name("math500")
        examples["MATH-500"] = [(item["problem"], item["answer"]) for item in math500[:5]]
    except Exception as e:
        warnings.append(f"MATH-500: {e}")

    try:
        from data.loader import load_dataset_by_name
        aime = load_dataset_by_name("aime2024")
        examples["AIME-2024"] = [(item["problem"], item["answer"]) for item in aime[:5]]
    except Exception as e:
        warnings.append(f"AIME-2024: {e}")

    return examples, warnings


dataset_examples, dataset_warnings = load_dataset_examples()

# Sidebar navigation
st.sidebar.title("导航")
page = st.sidebar.radio("选择页面", ["🧮 单题测试", "📊 实验结果", "📈 结果对比", "🚀 批量实验"])

if dataset_warnings:
    with st.sidebar.expander("数据集加载警告", expanded=False):
        for w in dataset_warnings:
            st.warning(w)
        st.info("提示: 如数据集加载失败，请先运行: python scripts/download_data.py")

if page == "🧮 单题测试":
    st.title("🧮 数学推理能力测试")
    st.markdown("基于 Qwen2.5-Math 和 DeepSeek-R1 的数学问题求解")

    # Sidebar settings
    st.sidebar.header("设置")
    model_choice = st.sidebar.selectbox(
        "选择模型",
        ["qwen2.5-math-1.5b", "deepseek-r1-qwen-1.5b"]
    )

    method_choice = st.sidebar.selectbox(
        "提示方法",
        ["cot", "self_refine", "self_consistency"]
    )

    # Load model button
    if "model" not in st.session_state:
        st.session_state.model = None
        st.session_state.tokenizer = None

    # OFFLINE MODE: Skip model loading if HF connection fails
    offline_mode = st.sidebar.checkbox("离线模式 (跳过模型加载)", value=False,
                                       help="如果无法连接 Hugging Face，启用此选项只查看已有结果")

    # Model loading with local cache support
    load_local = st.sidebar.checkbox("使用本地模型", value=True,
                                     help="使用已下载的本地模型，避免网络连接")

    if st.sidebar.button("加载模型"):
        try:
            with st.spinner("正在加载模型，请稍候..."):
                model, tokenizer = load_model(model_choice, use_local=load_local)
                st.session_state.model = model
                st.session_state.tokenizer = tokenizer
                st.sidebar.success("✅ 模型加载完成！")
        except Exception as e:
            st.sidebar.error(f"模型加载失败: {e}")
            st.session_state.model = None
            st.session_state.tokenizer = None

    # Main area
    st.header("输入数学问题")
    
    # Select dataset source
    source = st.selectbox("选择题目来源", ["自定义输入", "GSM8K", "MATH-500", "AIME-2024"])
    
    if source == "自定义输入":
        problem = st.text_area("输入你的数学问题", height=100)
        reference_answer = None
    else:
        examples = dataset_examples.get(source, [])
        if examples:
            example_options = [f"题目 {i+1}: {p[:80]}..." for i, (p, a) in enumerate(examples)]
            selected = st.selectbox(f"选择 {source} 题目", example_options)
            idx = example_options.index(selected)
            problem, reference_answer = examples[idx]
            
            st.markdown("**题目：**")
            st.markdown(problem)
            st.markdown(f"**标准答案：** {reference_answer}")
        else:
            st.error(f"无法加载 {source} 数据集")
            problem = ""
            reference_answer = None

    if st.button("求解", type="primary"):
        if st.session_state.model is None:
            st.error("请先加载模型！")
        elif not problem:
            st.warning("请输入数学问题")
        else:
            with st.spinner("正在思考..."):
                model = st.session_state.model
                tokenizer = st.session_state.tokenizer
                
                # Select method
                if method_choice == "cot":
                    response = solve_cot(model, tokenizer, problem)
                elif method_choice == "self_refine":
                    response = solve_self_refine(model, tokenizer, problem)
                else:  # self_consistency
                    response = solve_self_consistency(model, tokenizer, problem)
                
                # Display results
                st.header("📝 解答")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("提示方法", method_choice.upper())
                with col2:
                    st.metric("响应长度", f"{len(response)} 字符")
                
                st.markdown("### 推理过程")
                st.markdown(response)
                
                # Extract answer with dataset-specific handling
                from evaluation.metrics import extract_final_answer, normalize_answer
                
                # Determine dataset type for format-specific extraction
                dataset_type = source.lower().replace("-", "").replace(" ", "") if source != "自定义输入" else None

                answer = extract_final_answer(response, dataset_type)
                st.success(f"**模型提取的答案**: {answer}")

                # Compare with reference if available
                if reference_answer:
                    pred_norm = normalize_answer(extract_final_answer(response, dataset_type))
                    ref_norm = normalize_answer(extract_final_answer(reference_answer, dataset_type))
                    if pred_norm == ref_norm:
                        st.balloons()
                        st.success("✅ **答案正确！**")
                    else:
                        st.error(f"❌ **答案错误** (标准答案: {reference_answer})")

elif page == "📊 实验结果":
    st.title("📊 实验结果查看")
    st.markdown("查看已完成的实验结果")

    results_dir = "results"
    if not os.path.exists(results_dir):
        st.warning("暂无实验结果，请先运行实验！")
    else:
        result_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
        
        if not result_files:
            st.warning("results 目录为空，请先运行实验！")
        else:
            st.success(f"找到 {len(result_files)} 个实验结果文件")
            
            # Select result file
            selected_file = st.selectbox("选择结果文件", sorted(result_files))
            
            if selected_file:
                with open(os.path.join(results_dir, selected_file), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Display summary
                st.header("实验摘要")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("模型", data.get('model', 'N/A'))
                with col2:
                    st.metric("数据集", data.get('dataset', 'N/A'))
                with col3:
                    st.metric("方法", data.get('prompt_method', 'N/A'))
                
                # Display metrics
                metrics = data.get('metrics', {})
                st.header("评估指标")
                
                col1, col2 = st.columns(2)
                with col1:
                    accuracy = metrics.get('accuracy', 0)
                    st.metric("准确率", f"{accuracy:.2%}")
                with col2:
                    total = metrics.get('total_samples', 0)
                    st.metric("样本数", total)
                
                # Response length stats
                if 'response_length' in metrics:
                    rl = metrics['response_length']
                    st.subheader("响应长度统计")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("平均", f"{rl.get('char_mean', 0):.0f}")
                    with col2:
                        st.metric("中位数", f"{rl.get('char_median', 0):.0f}")
                    with col3:
                        st.metric("最小", rl.get('char_min', 0))
                    with col4:
                        st.metric("最大", rl.get('char_max', 0))
                
                # Show sample predictions
                with st.expander("查看详细预测结果"):
                    samples = data.get('samples', [])
                    for i, sample in enumerate(samples[:10]):  # Show first 10
                        st.markdown(f"**问题 {i+1}:** {sample.get('problem', 'N/A')[:100]}...")
                        st.markdown(f"- 预测: {sample.get('prediction', 'N/A')[:100]}...")
                        st.markdown(f"- 答案: {sample.get('reference', 'N/A')}")
                        st.markdown("---")

elif page == "📈 结果对比":
    st.title("📈 多实验对比")
    st.markdown("对比不同模型、数据集、方法的结果")
    
    results_dir = "results"
    if not os.path.exists(results_dir):
        st.warning("暂无实验结果")
    else:
        result_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
        
        if len(result_files) < 2:
            st.warning("需要至少2个实验结果才能对比")
        else:
            # Load all results
            all_results = []
            for f in result_files:
                with open(os.path.join(results_dir, f), 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    all_results.append({
                        'file': f,
                        'model': data.get('model', ''),
                        'dataset': data.get('dataset', ''),
                        'method': data.get('prompt_method', ''),
                        'accuracy': data.get('metrics', {}).get('accuracy', 0),
                        'total': data.get('metrics', {}).get('total_samples', 0)
                    })
            
            # Create comparison table
            import pandas as pd
            df = pd.DataFrame(all_results)
            
            st.header("实验结果对比表")
            st.dataframe(df[['model', 'dataset', 'method', 'accuracy', 'total']], use_container_width=True)
            
            # Visualization
            st.header("准确率对比")
            
            # Group by different dimensions
            group_by = st.selectbox("分组方式", ["model", "dataset", "method"])
            
            chart_data = df.groupby(group_by)['accuracy'].mean().reset_index()
            st.bar_chart(chart_data.set_index(group_by))

elif page == "🚀 批量实验":
    st.title("🚀 批量运行所有实验")
    st.markdown("一键运行全部18组实验配置")
    
    # Experiment configurations
    EXPERIMENTS = [
        # Qwen2.5-Math-1.5B experiments (1-9)
        {"id": 1, "model": "qwen2.5-math-1.5b", "dataset": "gsm8k", "method": "cot", "time": "~40min"},
        {"id": 2, "model": "qwen2.5-math-1.5b", "dataset": "gsm8k", "method": "self_refine", "time": "~80min"},
        {"id": 3, "model": "qwen2.5-math-1.5b", "dataset": "gsm8k", "method": "self_consistency", "time": "~200min"},
        {"id": 4, "model": "qwen2.5-math-1.5b", "dataset": "math500", "method": "cot", "time": "~15min"},
        {"id": 5, "model": "qwen2.5-math-1.5b", "dataset": "math500", "method": "self_refine", "time": "~30min"},
        {"id": 6, "model": "qwen2.5-math-1.5b", "dataset": "math500", "method": "self_consistency", "time": "~75min"},
        {"id": 7, "model": "qwen2.5-math-1.5b", "dataset": "aime2024", "method": "cot", "time": "~1min"},
        {"id": 8, "model": "qwen2.5-math-1.5b", "dataset": "aime2024", "method": "self_refine", "time": "~2min"},
        {"id": 9, "model": "qwen2.5-math-1.5b", "dataset": "aime2024", "method": "self_consistency", "time": "~5min"},
        # DeepSeek-R1-Qwen-1.5B experiments (10-18)
        {"id": 10, "model": "deepseek-r1-qwen-1.5b", "dataset": "gsm8k", "method": "cot", "time": "~40min"},
        {"id": 11, "model": "deepseek-r1-qwen-1.5b", "dataset": "gsm8k", "method": "self_refine", "time": "~80min"},
        {"id": 12, "model": "deepseek-r1-qwen-1.5b", "dataset": "gsm8k", "method": "self_consistency", "time": "~200min"},
        {"id": 13, "model": "deepseek-r1-qwen-1.5b", "dataset": "math500", "method": "cot", "time": "~15min"},
        {"id": 14, "model": "deepseek-r1-qwen-1.5b", "dataset": "math500", "method": "self_refine", "time": "~30min"},
        {"id": 15, "model": "deepseek-r1-qwen-1.5b", "dataset": "math500", "method": "self_consistency", "time": "~75min"},
        {"id": 16, "model": "deepseek-r1-qwen-1.5b", "dataset": "aime2024", "method": "cot", "time": "~1min"},
        {"id": 17, "model": "deepseek-r1-qwen-1.5b", "dataset": "aime2024", "method": "self_refine", "time": "~2min"},
        {"id": 18, "model": "deepseek-r1-qwen-1.5b", "dataset": "aime2024", "method": "self_consistency", "time": "~5min"},
    ]
    
    # Display experiment list
    st.header("实验配置列表 (共18组)")
    
    import pandas as pd
    df = pd.DataFrame(EXPERIMENTS)
    st.dataframe(df, use_container_width=True)
    
    # Calculate total time
    total_time = sum([int(e["time"].replace("~", "").replace("min", "")) for e in EXPERIMENTS])
    st.info(f"预计总时间: ~{total_time} 分钟 (~{total_time//60} 小时)")
    
    # Select experiments to run
    st.header("选择要运行的实验")
    
    col1, col2 = st.columns(2)
    with col1:
        run_qwen = st.checkbox("运行 Qwen 实验 (1-9)", value=True)
    with col2:
        run_deepseek = st.checkbox("运行 DeepSeek 实验 (10-18)", value=True)
    
    # Filter experiments
    selected_experiments = []
    if run_qwen:
        selected_experiments.extend([e for e in EXPERIMENTS if e["model"] == "qwen2.5-math-1.5b"])
    if run_deepseek:
        selected_experiments.extend([e for e in EXPERIMENTS if e["model"] == "deepseek-r1-qwen-1.5b"])
    
    if selected_experiments:
        selected_time = sum([int(e["time"].replace("~", "").replace("min", "")) for e in selected_experiments])
        st.success(f"已选择 {len(selected_experiments)} 个实验，预计时间: ~{selected_time} 分钟")
    
    # Run button
    st.header("开始运行")
    
    if st.button("🚀 一键运行选中实验", type="primary"):
        if not selected_experiments:
            st.error("请至少选择一个实验！")
        else:
            # Generate Python script
            script_content = f'''"""
批量运行实验脚本
Generated by Streamlit UI
"""
import subprocess
import sys

experiments = {selected_experiments}

print("="*70)
print("批量运行实验")
print("="*70)
print(f"总共 {{len(experiments)}} 个实验")
print()

for i, exp in enumerate(experiments, 1):
    print(f"\\n[{{i}}/{{len(experiments)}}] 运行实验 {{exp['id']}}: {{exp['model']}} + {{exp['dataset']}} + {{exp['method']}}")
    print(f"预计时间: {{exp['time']}}")
    
    # Use experiments/runner.py directly
    cmd = [
        sys.executable, "experiments/runner.py",
        "--model", exp["model"],
        "--dataset", exp["dataset"],
        "--method", exp["method"]
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ 完成")
        if result.stdout:
            print(result.stdout[-500:])  # Print last 500 chars of output
    except subprocess.CalledProcessError as e:
        print(f"❌ 失败: {{e}}")
        if e.stdout:
            print("输出:", e.stdout[-500:])
        if e.stderr:
            print("错误:", e.stderr[-500:])
        continue

print("\\n" + "="*70)
print("所有实验完成！")
print("="*70)
'''
            
            # Save script
            script_path = "run_all_experiments.py"
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(script_content)
            
            st.success(f"✅ 脚本已生成: `{script_path}`")
            st.code(f"python {script_path}", language="bash")
            
            st.info("💡 提示: 建议晚上挂机运行，避免占用工作时间")
            
            # Provide direct command option
            st.subheader("或直接运行命令")
            if run_qwen and not run_deepseek:
                st.code("python run_batch.py --model qwen2.5-math-1.5b", language="bash")
            elif run_deepseek and not run_qwen:
                st.code("python run_batch.py --model deepseek-r1-qwen-1.5b", language="bash")
            else:
                st.code("# 先运行 Qwen 实验\npython run_batch.py --model qwen2.5-math-1.5b\n\n# 再运行 DeepSeek 实验\npython run_batch.py --model deepseek-r1-qwen-1.5b", language="bash")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("CS6493 NLP Project")
st.sidebar.markdown("Topic 1: Mathematical Reasoning")
