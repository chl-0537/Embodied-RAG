#!/usr/bin/env python3
"""
语法错误修复脚本
用于修复rag_robot_framework.py中的缩进和语法问题
"""

import re

def fix_syntax_errors():
    """修复语法错误"""
    
    # 读取文件
    with open('rag_robot_framework.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 修复add_memory_entry_simple函数的缩进问题
    # 修复第256行的缩进
    content = re.sub(
        r'    try:\n        # 生成唯一ID\n    new_id = meta\.get\("id"\)',
        '    try:\n        # 生成唯一ID\n        new_id = meta.get("id")',
        content
    )
    
    # 修复第277行的缩进
    content = re.sub(
        r'        log_info\(f"Added memory entry to Chroma: \{new_id\}", chroma_metadata\)\n    return new_id\n\n    except Exception as e:',
        '        log_info(f"Added memory entry to Chroma: {new_id}", chroma_metadata)\n        return new_id\n        \n    except Exception as e:',
        content
    )
    
    # 修复call_chatglm_llm_json函数的缩进问题
    content = re.sub(
        r'            try:\n            parsed = json\.loads\(candidate\)',
        '            try:\n                parsed = json.loads(candidate)',
        content
    )
    
    content = re.sub(
        r'            if is_valid_json_structure\(parsed, schema\):\n                    return Result\(StatusCode\.OK, parsed, metadata=\{"attempt": attempt \+ 1, "raw_text": raw_text\}\)\n            else:\n                    log_warning\(f"Schema validation failed on attempt \{attempt\+1\}", parsed\)',
        '                if is_valid_json_structure(parsed, schema):\n                    return Result(StatusCode.OK, parsed, metadata={"attempt": attempt + 1, "raw_text": raw_text})\n                else:\n                    log_warning(f"Schema validation failed on attempt {attempt+1}", parsed)',
        content
    )
    
    # 修复call_chatglm_vision_json函数的缩进问题
    content = re.sub(
        r'    try:\n    content_list = \[\]',
        '    try:\n        content_list = []',
        content
    )
    
    content = re.sub(
        r'    content_list\.append\(\{"type": "image_url", "image_url": \{"url": image_url\}\}\)\n    content_list\.append\(\{"type": "text", "text": question\}\)\n    payload = \{',
        '        content_list.append({"type": "image_url", "image_url": {"url": image_url}})\n        content_list.append({"type": "text", "text": question})\n        payload = {',
        content
    )
    
    content = re.sub(
        r'        "model": VISION_MODEL,\n        "messages": \[\{"role": "user", "content": content_list\}\]\n    \}\n    resp = requests\.post\(CHATGLM_VISION_URL, json=payload, headers=HEADERS, timeout=60\)\n    data = resp\.json\(\)\n    raw_text = data\["choices"\]\[0\]\["message"\]\["content"\]',
        '            "model": VISION_MODEL,\n            "messages": [{"role": "user", "content": content_list}]\n        }\n        resp = requests.post(CHATGLM_VISION_URL, json=payload, headers=HEADERS, timeout=60)\n        data = resp.json()\n        raw_text = data["choices"][0]["message"]["content"]',
        content
    )
    
    content = re.sub(
        r'        try:\n        import re\n        m = re\.search\(r"\\{\.\*\\}", raw_text, re\.S\)\n            parsed_result = json\.loads\(m\.group\(0\)\) if m else \{"text": raw_text\}',
        '        try:\n            import re\n            m = re.search(r"\\{.*\\}", raw_text, re.S)\n            parsed_result = json.loads(m.group(0)) if m else {"text": raw_text}',
        content
    )
    
    # 写回文件
    with open('rag_robot_framework.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("语法错误修复完成！")

if __name__ == "__main__":
    fix_syntax_errors()
