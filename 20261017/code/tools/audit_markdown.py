"""Inventory all local Markdown, verify local links and preserved pinned documents.

Does not access the network or modify source documentation. Generated audit
files are stored under results/documentation_audit. Exit nonzero on broken
local links, damaged UTF-8/question-mark text, or pinned-byte changes.
"""
from pathlib import Path
from urllib.parse import unquote, urlsplit
import hashlib
import json
import os
import re
import argparse

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT.parent
OUT = PROJECT/'results/documentation_audit'
PATTERN = re.compile(r'\[([^\]]*)\]\((<[^>]+>|[^)]+)\)')


def classify(relative):
    if relative.endswith('PROJECT_MAP.md') or relative.endswith('INVENTORY_ZH.md'):
        return '自动索引', '自动生成，不手改'
    if '/documentation_audit/' in relative:
        return '文档治理', '本轮文件组织/链接核查'
    if '/code/docs/' in relative:
        if relative.endswith('/README.md'):
            return '方法导航', '当前方法索引'
        return '方法/协议', '阶段及固定快照见code/docs/README'
    if relative.endswith('feasibility review.md'):
        return '项目可行性', '保留用户指定英文文件名；含历史阶段说明'
    if '/experiments/improved_3d_v4/' in relative:
        return 'v4独立实验', '训练已提交；审计已完成；模型结果待核实'
    if '/experiments/improved_3d_v3/' in relative:
        return 'v3模型实验', '训练完成；跨模型比较位于model_evaluation'
    if '/model_evaluation/' in relative:
        return '跨模型性能与泛化评估', '统一best选模；课程指标与补充指标分区'
    if '/experiments/improved_3d_v2/' in relative:
        return 'v2独立实验', '训练完成；方案与结果分列'
    if '/data_analysis/' in relative:
        return 'Train数据审计', '图片/审计说明就近存放'
    if '/diagnostics/' in relative:
        return '固定权重诊断', '图片就近解读；正控失败不做因果归因'
    if '/case_studies/' in relative:
        return '分割案例', '按split/主题组织；历史Standard/v1为主'
    if '/figures/' in relative or '/tables/' in relative:
        return '历史图表/指标', '保留来源；最新跨版本比较另见v3/analysis'
    if '/results/reports/' in relative:
        return '综合研究/治理', '研究过程材料；不是实时执行命令'
    if '/logs/' in relative:
        return '原始记录说明', '不改写原始实验JSON'
    return '项目/目录入口', '导航及分类规则'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory',action='store_true',help='Optionally generate a separate Markdown inventory; PROJECT_MAP is the default file index')
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    docs = sorted(p for p in ROOT.rglob('*.md') if not set(p.parts)&{'.git','node_modules','.venv','venv'})
    broken, damaged, records = [], [], []
    protected = {}
    snapshot = OUT/'before_20261020.json'
    if snapshot.exists():
        protected = json.loads(snapshot.read_text(encoding='utf-8'))['protected']
    registry=PROJECT/'code/config/artifact_paths.json'
    migration=json.loads(registry.read_text(encoding='utf-8')) if registry.exists() else {}
    def current_path(name):
        prefix='20261017/'
        key=name[len(prefix):] if name.startswith(prefix) else name
        if key in migration.get('file_renames',{}):
            return PROJECT/migration['file_renames'][key]
        for old,new in migration.get('directory_renames',{}).items():
            if key==old or key.startswith(old+'/'):return PROJECT/(new+key[len(old):])
        return ROOT/name
    ledger=OUT/'consolidation_20261020.json'
    approved=json.loads(ledger.read_text(encoding='utf-8')).get('approved_protected_retirements',{}) if ledger.exists() else {}
    changed = [name for name,h in protected.items()
               if not ((not current_path(name).exists() and approved.get(name)==h)
                       or (current_path(name).exists() and hashlib.sha256(current_path(name).read_bytes()).hexdigest()==h))]
    protected_current={current_path(name).resolve() for name in protected if current_path(name).exists()}
    for path in docs:
        relative = path.relative_to(ROOT).as_posix()
        text = path.read_text(encoding='utf-8-sig')
        prose = re.sub(r'```.*?```','',text,flags=re.S)
        headings = re.findall(r'^#\s+(.+)$',prose,flags=re.M)
        if '\ufffd' in text or re.search(r'\?{4,}',prose):
            damaged.append(relative)
        for match in PATTERN.finditer(prose):
            target = match.group(2).strip('<>')
            if urlsplit(target).scheme or target.startswith('#'):
                continue
            local = unquote(target.split('#')[0])
            if local and not (path.parent/local).exists():
                broken.append({'file':relative,'target':target})
        category,status = classify(relative)
        records.append({'path':relative,'title':headings[0] if headings else '(无一级标题)',
                        'h1_count':len(headings),'category':category,'status':status,
                        'protected':path.resolve() in protected_current,
                        'sha256':None if path.name=='INVENTORY_ZH.md' else hashlib.sha256(path.read_bytes()).hexdigest()})
    result = {'markdown_count':len(records),'local_link_scope':'Markdown inline relative links; excludes fenced code, URLs and fragment validation',
              'broken_local_links':broken,'encoding_flags':damaged,
              'protected_count':len(protected),'approved_protected_retirements':approved,
              'changed_protected_files':changed,'files':records}
    (OUT/'audit_20261020.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['# 全部Markdown分类清单','','自动生成；每个文件的类别、标题、位置与阶段用途如下。历史README及哈希锁定文件保留原名，当前状态从对应索引读取。','',
           '| 文档 | 标题 | 分类 | 阶段与用途 | 固定字节 |','|---|---|---|---|---|']
    for row in records:
        p=ROOT/row['path']; rel=os.path.relpath(p,OUT).replace('\\','/')
        if ' ' in rel: rel='<'+rel+'>'
        lines.append(f"| [{p.name}]({rel}) | {row['title'].replace('|',' / ')} | {row['category']} | {row['status']} | {'是' if row['protected'] else '否'} |")
    lines += ['',f"统计：{len(records)}份Markdown；断链{len(broken)}；编码疑点{len(damaged)}；固定文档变化{len(changed)}。",'',
              '检查不判断医疗结论正确性，不验证外部网站或标题锚点，不把文档数量当实验完成度。','']
    if args.inventory:
        (OUT/'INVENTORY_ZH.md').write_text('\n'.join(lines),encoding='utf-8')
    print(f'Markdown={len(records)}; broken_links={len(broken)}; encoding_flags={len(damaged)}; protected_changed={len(changed)}')
    if broken or damaged or changed:
        print(json.dumps({'broken':broken,'encoding':damaged,'protected':changed},ensure_ascii=False,indent=2))
        raise SystemExit(1)


if __name__=='__main__':
    main()
