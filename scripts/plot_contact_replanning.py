"""Render the measured trajectories with ReportLab's standard LinePlot chart.

Runs in the existing bundled artifact runtime; changes no robotics dependencies.
"""
import argparse
import json
from pathlib import Path
from reportlab.graphics.shapes import Drawing, String, Line, Rect
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics import renderSVG
from reportlab.lib.colors import HexColor


def read(p):
    return json.loads(p.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    out = ap.parse_args().output
    data = read(out/'analysis.json')
    d = Drawing(1100, 820)
    d.add(Rect(0,0,1100,820,fillColor=HexColor('#ffffff'),strokeColor=None))
    d.add(String(40,790,'CloseFridge: per-door opening and replanning events',fontSize=22))
    d.add(String(40,765,'Blue: fixed H=50    Orange: event-triggered    Vertical dashed: candidate events',fontSize=14))
    for panel,seed in enumerate((43,44,45,46)):
        x,y = 65+(panel%2)*545, 445-(panel//2)*355
        if seed not in data['status']['valid_seeds']:
            d.add(String(x,y+275,f'Seed {seed}: incomplete',fontSize=17))
            d.add(String(x,y+150,'Control stopped on CUDA OOM before inference/action.',fontSize=13))
            d.add(String(x,y+125,'Candidate not run. No trajectory or success result.',fontSize=13))
            continue
        chart = LinePlot()
        chart.x,chart.y,chart.width,chart.height=x,y,450,250
        curves=[]
        styles=[]
        for arm,color in [('baseline','#1769aa'),('candidate','#df6518')]:
            p=Path(__file__).resolve().parents[1]/data['metrics'][f'{arm}-{seed}']['path']
            reset=read(p/'reset.json')
            rs=[json.loads(s) for s in (p/'steps.jsonl').read_text().splitlines()]
            import math
            for door,k in enumerate(reset['fridge']['qpos']):
                curves.append([(i,math.degrees(r['fridge']['qpos'][k])) for i,r in enumerate([reset]+rs)])
                styles.append((color,door))
        chart.data=curves
        chart.xValueAxis.valueMin=0
        chart.xValueAxis.valueMax=600
        chart.xValueAxis.valueStep=150
        chart.yValueAxis.valueMin=-5
        chart.yValueAxis.valueMax=95
        chart.yValueAxis.valueSteps=[0,30,60,90]
        for i,(color,door) in enumerate(styles):
            chart.lines[i].strokeColor=HexColor(color)
            chart.lines[i].strokeWidth=1.4
            if door:
                chart.lines[i].strokeDashArray=[6,3]
        d.add(chart)
        d.add(String(x,y+275,f'Seed {seed}',fontSize=17))
        d.add(String(x+140,y-35,'Executed control actions',fontSize=12))
        d.add(String(x-45,y+260,'Degrees',fontSize=11))
        for event in data['events'][f'candidate-{seed}']:
            xx=x+event['step']/600*450
            d.add(Line(xx,y,xx,y+250,strokeColor=HexColor('#a33b13'),strokeWidth=.7,strokeDashArray=[3,4]))
    d.add(String(40,25,'Angles are evaluation only. Dashed curves denote a second door. Traces end at episode termination.',fontSize=12))
    renderSVG.drawToFile(d,str(out/'door-progress.svg'))


if __name__ == '__main__':
    main()
