#!/usr/bin/env python3
"""
Markdown文件整理脚本 - 只保留最重要的文档在根目录
"""

import os
import shutil
from pathlib import Path

def organize_markdown_files():
    """整理markdown文件，保留重要文档，归档其他文件"""
    
    root_dir = Path("/home/coder-gw/7Projects_in_7Days/Layerwise-Adapter")
    archived_dir = root_dir / "archived_versions"
    
    print("📋 开始整理Markdown文件...")
    
    # 确保归档目录存在
    (archived_dir / "old_docs").mkdir(exist_ok=True)
    
    # ======== 保留在根目录的核心文档 ========
    keep_in_root = {
        "README.md": "项目主文档",
        "USAGE_GUIDE.md": "使用指南", 
        "PROJECT_STRUCTURE_FINAL.md": "最终项目结构",
        "PAPER_PUBLICATION_CHECKLIST.md": "论文发表检查清单",
        "PAPER_CRITICAL_CORRECTIONS.md": "论文关键修正",
    }
    
    print("\n✅ 保留在根目录的核心文档:")
    for filename, desc in keep_in_root.items():
        filepath = root_dir / filename
        if filepath.exists():
            print(f"   📄 {filename} - {desc}")
        else:
            print(f"   ❌ {filename} - 不存在")
    
    # ======== 需要归档的根目录markdown文件 ========
    root_md_files = list(root_dir.glob("*.md"))
    
    print(f"\n📦 归档根目录多余的markdown文件:")
    for md_file in root_md_files:
        if md_file.name not in keep_in_root:
            new_location = archived_dir / "old_docs" / md_file.name
            try:
                shutil.move(str(md_file), str(new_location))
                print(f"   📦 {md_file.name} -> archived_versions/old_docs/")
            except Exception as e:
                print(f"   ❌ 移动 {md_file.name} 失败: {e}")
    
    # ======== 整理paper目录 ========
    paper_dir = root_dir / "paper"
    if paper_dir.exists():
        # 保留的paper文档
        keep_in_paper = {
            "abstract.md": "论文摘要",
            "updated_comprehensive_paper.md": "完整论文",
            "references.bib": "参考文献",
        }
        
        print(f"\n📁 整理paper目录:")
        paper_files = list(paper_dir.glob("*.md"))
        for paper_file in paper_files:
            if paper_file.name not in keep_in_paper:
                new_location = archived_dir / "old_docs" / f"paper_{paper_file.name}"
                try:
                    shutil.move(str(paper_file), str(new_location))
                    print(f"   📦 paper/{paper_file.name} -> archived_versions/old_docs/")
                except Exception as e:
                    print(f"   ❌ 移动 {paper_file.name} 失败: {e}")
    
    # ======== 整理docs目录 ========
    docs_dir = root_dir / "docs"
    if docs_dir.exists():
        print(f"\n📁 归档docs目录中的文档:")
        docs_files = list(docs_dir.glob("*.md"))
        for doc_file in docs_files:
            new_location = archived_dir / "old_docs" / f"docs_{doc_file.name}"
            try:
                shutil.move(str(doc_file), str(new_location))
                print(f"   📦 docs/{doc_file.name} -> archived_versions/old_docs/")
            except Exception as e:
                print(f"   ❌ 移动 {doc_file.name} 失败: {e}")
    
    print(f"\n✅ Markdown文件整理完成!")
    print(f"📄 根目录保留: {len(keep_in_root)}个核心文档")
    print(f"📦 已归档: 大量过时文档到 archived_versions/old_docs/")

if __name__ == "__main__":
    organize_markdown_files()
