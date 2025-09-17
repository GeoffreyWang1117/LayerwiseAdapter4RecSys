#!/bin/bash

# WWW2026论文编译脚本
# 使用pdflatex编译LaTeX论文

echo "🚀 开始编译WWW2026论文..."
echo "================================"

# 检查LaTeX是否安装
if ! command -v pdflatex &> /dev/null; then
    echo "❌ 错误: pdflatex未安装"
    echo "请安装TeX Live或MiKTeX"
    exit 1
fi

# 检查bibtex是否安装
if ! command -v bibtex &> /dev/null; then
    echo "❌ 错误: bibtex未安装"
    exit 1
fi

# 进入paper目录
cd "$(dirname "$0")"

# 清理之前的编译文件
echo "🧹 清理编译文件..."
rm -f *.aux *.bbl *.blg *.log *.out *.toc *.fdb_latexmk *.fls *.synctex.gz

# 第一次编译
echo "📝 第一次pdflatex编译..."
pdflatex -interaction=nonstopmode www2026_paper.tex

# 编译参考文献
echo "📚 编译参考文献..."
bibtex www2026_paper

# 第二次编译（处理引用）
echo "📝 第二次pdflatex编译..."
pdflatex -interaction=nonstopmode www2026_paper.tex

# 第三次编译（确保所有引用正确）
echo "📝 第三次pdflatex编译..."
pdflatex -interaction=nonstopmode www2026_paper.tex

# 检查编译结果
if [ -f "www2026_paper.pdf" ]; then
    echo "✅ 论文编译成功！"
    echo "📄 输出文件: www2026_paper.pdf"
    
    # 显示PDF信息
    if command -v pdfinfo &> /dev/null; then
        echo ""
        echo "📊 PDF信息:"
        pdfinfo www2026_paper.pdf | grep -E "Pages|Creator|Producer"
    fi
    
    # 显示文件大小
    echo "📏 文件大小: $(ls -lh www2026_paper.pdf | awk '{print $5}')"
    
else
    echo "❌ 编译失败！"
    echo "请检查LaTeX日志文件: www2026_paper.log"
    exit 1
fi

echo ""
echo "🎉 编译完成！"
echo "可以使用PDF阅读器打开 www2026_paper.pdf 查看论文"
