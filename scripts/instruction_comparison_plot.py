"""Static comparison from completed evidence, using bundled ReportLab; no model imports."""
import argparse
import json
from pathlib import Path
from reportlab.graphics.shapes import Drawing,String,Rect
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics import renderSVG,renderPDF
from reportlab.lib import colors

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results-directory',type=Path,required=True)
    run=ap.parse_args().results_directory
    rows=json.loads((run/'results.json').read_text())
    assert len(rows)<=60 and all(r['valid'] for r in rows)
    complete=(run/'complete.json').exists()
    if complete:assert json.loads((run/'complete.json').read_text())['episodes']==60
    else:assert (run/'stopped.json').exists()
    drawing=Drawing(1080,430)
    drawing.add(Rect(0,0,1080,430,fillColor=colors.white,strokeColor=None))
    drawing.add(String(32,397,'Instruction-controlled CloseFridge evaluation',fontName='Helvetica-Bold',fontSize=22))
    subtitle='60 valid cells; 10 matched diagnostic scenes' if complete else f'INCOMPLETE: {len(rows)}/60 valid cells. Rates below use completed cells only; denominators differ.'
    drawing.add(String(32,373,subtitle,fontSize=12,fillColor=colors.HexColor('#475569')))
    for i,(title,doors) in enumerate([('All scenes',None),('Single-door scenes',1),('Double-door scenes',2)]):
        selected=[r for r in rows if doors is None or r['door_count']==doors]
        counts=[[sum(r['model']==m and r['condition']==c for r in selected) for m in 'ABC'] for c in ['ID','NL']]
        chart=VerticalBarChart();chart.x=55+i*350;chart.y=120;chart.width=285;chart.height=195
        chart.data=[[100*sum(r['success'] for r in selected if r['model']==m and r['condition']==c)/counts[j][k] for k,m in enumerate('ABC')] for j,c in enumerate(['ID','NL'])]
        chart.categoryAxis.categoryNames=['A','B','C'];chart.categoryAxis.labels.fontSize=13
        chart.valueAxis.valueMin=0;chart.valueAxis.valueMax=100;chart.valueAxis.valueStep=25
        chart.valueAxis.labelTextFormat='%d%%';chart.valueAxis.labels.fontSize=9
        chart.bars[0].fillColor=colors.HexColor('#1d4ed8');chart.bars[1].fillColor=colors.HexColor('#d97706')
        chart.bars.strokeColor=None;chart.groupSpacing=14;chart.barSpacing=3
        chart.barLabelFormat='%0.1f';chart.barLabels.fontSize=10
        drawing.add(chart)
        drawing.add(String(55+i*350,338,title,fontName='Helvetica-Bold',fontSize=14))
        drawing.add(String(55+i*350,82,'n (A/B/C): ID '+ '/'.join(map(str,counts[0]))+'; NL '+ '/'.join(map(str,counts[1])),fontSize=9))
    drawing.add(Rect(32,63,12,12,fillColor=colors.HexColor('#1d4ed8'),strokeColor=None))
    drawing.add(String(51,64,'Task-ID: CloseFridge',fontSize=12))
    drawing.add(Rect(260,63,12,12,fillColor=colors.HexColor('#d97706'),strokeColor=None))
    drawing.add(String(279,64,'Native natural-language instruction',fontSize=12))
    drawing.add(String(32,36,'A: pretrained    B: 300-update NL fine-tuning    C: 300-update task-ID fine-tuning',fontSize=12))
    drawing.add(String(32,15,'Success is the simulator task predicate. Small diagnostic sample; bars do not establish generalization.',fontSize=10,fillColor=colors.HexColor('#475569')))
    renderSVG.drawToFile(drawing,str(run/'comparison-plot.svg'))
    import io
    import pypdfium2 as pdfium
    buffer=io.BytesIO()
    renderPDF.drawToFile(drawing,buffer)
    pdf=pdfium.PdfDocument(buffer.getvalue())
    page=pdf[0]
    bitmap=page.render(scale=2)
    bitmap.to_pil().save(run/'comparison-plot.png')
    bitmap.close();page.close();pdf.close()
    print(run/'comparison-plot.svg')

if __name__=='__main__':main()
