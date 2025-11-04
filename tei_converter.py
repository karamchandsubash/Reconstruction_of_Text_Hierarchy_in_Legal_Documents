import json
from pathlib import Path
from lxml import etree
IN_FILE = 'labelstudio_tasks_with_text.json'
OUT_DIR = Path('tei_output'); OUT_DIR.mkdir(exist_ok=True)
mapping = {
    'Title': ('div', {'type': 'title'}, 'head'),
    'Paragraph': ('p', {}, None),
    'Section-Heading': ('div', {'type': 'section'}, 'head'),
    'List-Item': ('list', {}, 'item'),
    'Page-Footer': ('note', {'type': 'page-footer'}, None),
    'Page-Header': ('note', {'type': 'page-header'}, None),
    'Page-Number': ('pb', {}, None),
    'Caption': ('figure', {}, 'figDesc'),
    'Picture': ('figure', {}, 'figDesc')
}
with open(IN_FILE, 'r', encoding='utf-8') as f:
    tasks = json.load(f)
for t in tasks:
    image = t['data']['image']; name = Path(image).stem
    TEI = etree.Element('TEI', nsmap={None:'http://www.tei-c.org/ns/1.0'})
    text_el = etree.SubElement(TEI, 'text'); body = etree.SubElement(text_el, 'body')
    ann = t.get('annotations', [{}])[0].get('result', [])
    ann = sorted(ann, key=lambda r: r['value'].get('y', 0))
    for r in ann:
        val = r['value']; lab = val['rectanglelabels'][0]; content = val.get('text','')
        tag, attrs, sub = mapping.get(lab, ('p', {}, None))
        el = etree.SubElement(body, tag, **attrs)
        if sub:
            se = etree.SubElement(el, sub); se.text = content
        else:
            el.text = content
    xml_bytes = etree.tostring(TEI, pretty_print=True, encoding='utf-8', xml_declaration=True)
    with open(OUT_DIR / f'{name}.xml', 'wb') as fo:
        fo.write(xml_bytes)
print('TEI conversion done ->', OUT_DIR)
