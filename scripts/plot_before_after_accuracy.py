"""Plot measured original-versus-selected HMM scores; no training or estimation."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT/'outputs/experiments'
student = json.loads((ARCHIVE/'hmm_tuning_001/student_validation.json').read_text())
original = json.loads((ARCHIVE/'evaluator_test_rf_hmm_001/metrics.json').read_text())
selected = json.loads((ARCHIVE/'evaluator_test_rf_hmm_tuned_001/metrics.json').read_text())
assert original['known_observations'] == selected['known_observations']
before = np.array([student['original_hmm']['pooled_top1'], original['hmm_top1_accuracy']])*100
after = np.array([student['selected_hmm']['pooled_top1'], selected['hmm_top1_accuracy']])*100
x = np.arange(2)
fig, ax = plt.subplots(figsize=(10,6.4), facecolor='white')
ax.set_axisbelow(True)
ax.grid(axis='y', color='#e6ebef', linewidth=.8)
first = ax.bar(x-.18, before, .32, color='#aebbc7', label='Before: original HMM')
second = ax.bar(x+.18, after, .32, color='#d99131', label='After: tuned HMM')
for bars in [first,second]:
    ax.bar_label(bars, labels=[f'{bar.get_height():.2f}%' for bar in bars], padding=7, fontsize=13, fontweight='bold', color='#15324d')
ax.set_ylim(0,108)
ax.set_yticks(np.arange(0,101,20))
ax.set_ylabel('Pointwise accuracy (%)', fontsize=12)
ax.set_xticks(x, ['Student test','Repeat evaluator benchmark'], fontsize=12)
ax.tick_params(axis='both',length=0,pad=10)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.legend(loc='upper center',bbox_to_anchor=(.5,1.16),frameon=False,ncol=2,fontsize=10)
fig.text(.07,.955,'HMM/Viterbi: before and after tuning',fontsize=21,fontweight='bold',color='#15324d')
fig.text(.07,.908,'Same frozen Random Forest. Only the decoder settings changed.',fontsize=11,color='#566679')
fig.text(.07,.09,f'Student test: +{after[0]-before[0]:.2f} percentage points   |   Evaluator: +{after[1]-before[1]:.2f} percentage points',fontsize=11,color='#15324d')
fig.text(.07,.047,'Accuracy against valid derived point references. Decoder selected using training OOF scores.',fontsize=9,color='#566679')
fig.text(.07,.020,'Evaluator data had already been examined; its repeat score is a development benchmark.',fontsize=9,color='#566679')
fig.subplots_adjust(left=.09,right=.97,bottom=.20,top=.76)
for extension in ['png','svg']:
    destination=ARCHIVE/f'hmm_tuning_001/before_after_accuracy.{extension}'
    fig.savefig(destination,dpi=200,facecolor='white')
    if extension == 'svg':
        destination.write_text('\n'.join(line.rstrip() for line in destination.read_text().splitlines())+'\n')
    print(destination)
plt.close(fig)
