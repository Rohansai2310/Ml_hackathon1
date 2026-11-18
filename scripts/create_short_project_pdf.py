"""Create a three-page plain-language explanation from saved project evidence."""
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.platypus import PageBreak, Spacer

from create_full_project_pdf import GuideDoc, STYLES, para, table, WIDTH, HEIGHT

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/PathSnap_Simple_Project_Explanation.pdf'
ARCHIVE = ROOT / 'outputs/experiments'


class ShortGuide(GuideDoc):
    def decorate(self, canvas, doc):
        canvas.saveState()
        canvas.setFillColor(colors.HexColor('#15324d'))
        canvas.rect(0, HEIGHT-32, WIDTH, 32, fill=1, stroke=0)
        canvas.setFont('BodyBold', 8)
        canvas.setFillColor(colors.white)
        canvas.drawString(43, HEIGHT-21, 'PATHSNAP | SIMPLE PROJECT EXPLANATION')
        canvas.setFont('Body', 8)
        canvas.setFillColor(colors.HexColor('#566679'))
        canvas.drawString(43, 26, '18 November 2026 | Random Forest + HMM/Viterbi')
        canvas.drawRightString(WIDTH-43, 26, f'{doc.page} / 3')
        canvas.restoreState()


def main():
    current = json.loads((ARCHIVE/'evaluator_test_rf_hmm_tuned_001/metrics.json').read_text())
    old = json.loads((ARCHIVE/'evaluator_test_rf_hmm_001/metrics.json').read_text())
    student = json.loads((ARCHIVE/'hmm_tuning_001/student_validation.json').read_text())
    STYLES['Text'].fontSize = 10.5
    STYLES['Text'].leading = 16
    STYLES['Text'].spaceAfter = 10
    STYLES['Note'].fontSize = 9
    STYLES['Note'].leading = 13
    story = []

    def h(text):
        story.append(para(text, 'Sub'))

    def p(text):
        story.append(para(text))

    story.append(para('1. What our project does', 'Chapter'))
    p('PathSnap finds the road a vehicle most likely travelled using its GPS points. GPS can be inaccurate: a point may appear between roads or closer to the wrong road. Our system uses the nearby roads, travel direction and the complete trip to make a better match.')
    h('The complete flow')
    story.append(table(['Input', 'Processing', 'Output'], [[
        'GPS points + road network', 'Nearby roads → 22 features → Random Forest → HMM/Viterbi',
        'Matched roads and connected route segments',
    ]], [125, 230, WIDTH-441]))
    h('Step 1: Read the data')
    p('Each trip has GPS coordinates and timestamps, road junctions, directed road links, and a reference route. A directed link, called an arc, represents one permitted direction of travel. Reference routes are used for learning and checking results; prediction does not read them.')
    h('Step 2: Find possible roads')
    p('We convert coordinates into metres and find roads within 75 metres of each GPS point. We keep up to 40 nearby road candidates. This gives the model a small set of possible answers for each point.')
    h('Step 3: Describe each candidate')
    p('We calculate 22 features. These describe distance to the road, travel direction, speed, nearby GPS points and road connections. For example, a slightly farther road may be better if its direction agrees with the vehicle’s movement.')
    h('How the data is separated')
    p('There are 80 training trips, 10 student-test trips and 10 evaluator-test trips. Two training trips with unreliable reference coverage are excluded, leaving 78 for fitting. Validation keeps whole trips together so neighbouring points from one trip do not appear on both sides of a split.')

    story.extend([PageBreak(), para('2. How the model finds the route', 'Chapter')])
    h('Random Forest: score each possible road')
    p('Random Forest combines 200 decision trees. It learns from examples of correct and incorrect candidate roads. For a new GPS point, it gives every candidate a score indicating how likely it is to be the correct road.')
    p('Several classifiers were compared during development. We selected Random Forest. The running system uses this one classifier; we do not need all the comparison models in either client.')
    h('Why Random Forest alone is not enough')
    p('Choosing the highest score separately for every point can make the route jump between parallel roads or disconnected streets. A good local answer does not always produce a sensible whole trip.')
    h('HMM: connect the road choices')
    p('The HMM uses two kinds of evidence: the Random Forest score for each candidate, and whether movement between consecutive candidates makes sense on the directed road network. It compares road travel distance with the movement seen in GPS.')
    h('Viterbi: choose the best complete sequence')
    p('Viterbi finds the highest-scoring sequence across the trip. For example, one GPS point may be closer to a side street. If the surrounding points and road connections support the main road, Viterbi can keep the vehicle on the main road instead.')
    h('What we fixed and improved')
    p('We reviewed directed paths, distances in metres, movement partway along a road, missing candidates and gaps. If the trip cannot be connected, the decoder returns separate segments instead of inventing a connection.')
    p('We also tested 25 decoder configurations using training predictions. The chosen settings allow up to 15 metres of small backwards GPS projection noise on the same road, treating it as staying still. This does not allow driving against a one-way road. Settings were frozen before student validation and the repeat evaluator check.')

    story.extend([PageBreak(), para('3. Results and two-laptop setup', 'Chapter')])
    h('Measured pointwise accuracy')
    story.append(table(['Pipeline / dataset', 'Accuracy'], [
        ['Random Forest alone — evaluator', f"{current['top1_accuracy']:.2%}"],
        ['Random Forest + original HMM — evaluator', f"{old['hmm_top1_accuracy']:.2%}"],
        ['Random Forest + improved HMM — repeat evaluator', f"{current['hmm_top1_accuracy']:.2%}"],
        ['Random Forest + improved HMM — student test', f"{student['selected_hmm']['pooled_top1']:.2%}"],
    ], [WIDTH-186, 100]))
    story.append(Spacer(1, 10))
    p(f"The improved pipeline correctly matched {current['hmm_correct_known_observations']:,} of {current['known_observations']:,} GPS points with valid derived references. Compared with the original HMM, accuracy increased by {(current['hmm_top1_accuracy']-old['hmm_top1_accuracy'])*100:.2f} percentage points.")
    p('Accuracy here measures the road chosen for each GPS point, not whether an entire route is perfect. Labels are derived from reference routes; 21 evaluator points have unresolved references and are excluded. This evaluator set had already been examined, so 90.79% is a repeat development benchmark. Its results were not used to select the improved settings.')
    h('How the two-laptop architecture works')
    p('Your laptop runs FastAPI, one trained Random Forest, HMM/Viterbi and evaluation. The other laptop runs the interface. On the same Wi-Fi, it sends a selected trip to your server IP and receives the route and results. No model or raw dataset is needed on the UI laptop.')
    story.append(table(['Laptop 1: server', 'Wi-Fi', 'Laptop 2: interface'], [[
        'Random Forest + HMM + evaluation', 'Request → response', 'Select trip, display map and accuracy',
    ]], [190, 110, WIDTH-386]))
    h('What is ready')
    p('The server and interface are implemented and tested as separate local processes. The UI lists all ten trips, displays GPS/reference/predicted roads, shows accuracy and lets you download results. API access through the server Wi-Fi address was checked. The remaining demonstration step is to connect a second physical laptop using the commands in docs/CLIENT_API.md.')

    doc = ShortGuide(OUT)
    doc.title = 'PathSnap: Simple Project Explanation'
    doc.build(story)
    if doc.page != 3:
        raise RuntimeError(f'Expected three pages, got {doc.page}')
    print(f'Created {OUT} ({doc.page} pages)')


if __name__ == '__main__':
    main()
