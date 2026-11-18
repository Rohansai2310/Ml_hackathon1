"""Build the project handoff PDF from recorded local evidence; no model fitting."""
from pathlib import Path
import csv
import textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/PathSnap_Project_Progress_and_HMM_Guide.pdf'
NAVY, TEAL, INK, MUTED = '#142c46', '#087e86', '#26374a', '#617286'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42})

class Page:
    def __init__(self, pdf, number, title, subtitle):
        self.pdf, self.number = pdf, number
        self.fig = plt.figure(figsize=(8.27, 11.69), facecolor='white')
        self.ax = self.fig.add_axes([0, 0, 1, 1]); self.ax.set_axis_off(); self.ax.set_xlim(0, 1); self.ax.set_ylim(0, 1)
        self.ax.axhspan(.955, 1, color=NAVY)
        self.ax.text(.075,.974,'PATHSNAP  /  PROJECT HANDOFF',color='white',fontsize=9,weight='bold',va='center')
        self.ax.text(.075,.918,title,color=NAVY,fontsize=23,weight='bold',va='top')
        self.ax.text(.075,.871,subtitle,color=MUTED,fontsize=10,va='top')
        self.y = .823
    def h(self, title):
        self.ax.text(.075,self.y,title,color=TEAL,fontsize=13,weight='bold',va='top')
        self.y -= .030
    def p(self, text, size=9.8):
        lines = textwrap.wrap(text, width=92 if size <= 10.4 else 84, break_long_words=False, break_on_hyphens=False)
        self.ax.text(.075,self.y,'\n'.join(lines),fontsize=size,color=INK,va='top',linespacing=1.4)
        self.y -= len(lines)*.0167 + .012
    def bullet(self, text):
        self.p('•  ' + text)
    def code(self, text):
        lines=text.splitlines()
        height=len(lines)*.018+.025
        self.ax.add_patch(FancyBboxPatch((.075,self.y-height+.007),.85,height,boxstyle='round,pad=.009',facecolor='#f0f4f8',edgecolor='none'))
        self.ax.text(.09,self.y,text,fontsize=8.5,family='DejaVu Sans Mono',va='top',color=NAVY,linespacing=1.4)
        self.y-=height+.022
    def table(self, headers, rows, widths=None):
        height=.034*(len(rows)+1)
        t=self.ax.table(cellText=rows,colLabels=headers,cellLoc='left',colLoc='left',bbox=[.075,self.y-height,.85,height],colWidths=widths)
        t.auto_set_font_size(False); t.set_fontsize(9)
        for (r,c),cell in t.get_celld().items():
            cell.set_edgecolor('white'); cell.PAD=.08
            cell.set_facecolor(NAVY if r==0 else ('#edf3f7' if r%2 else '#f7f9fb'))
            cell.get_text().set_color('white' if r==0 else INK)
            if r==0: cell.get_text().set_weight('bold')
        self.y-=height+.027
    def end(self):
        if self.y < .065:
            raise RuntimeError(f'Page {self.number} overflows: y={self.y}')
        self.ax.plot([.075,.925],[.057,.057],color='#dbe3eb',lw=1)
        self.ax.text(.075,.034,'18 November 2026  •  model-training  •  baseline 0d1759d',fontsize=8,color=MUTED)
        self.ax.text(.925,.034,str(self.number),ha='right',fontsize=8,color=MUTED)
        self.pdf.savefig(self.fig); plt.close(self.fig)

with PdfPages(OUT) as pdf:
    pdf.infodict().update(Title='PathSnap — Project Progress and HMM/Viterbi Guide',Author='PathSnap project team',Subject='Verified implementation, recorded model results and next-stage architecture')
    p=Page(pdf,1,'What is done, and what comes next','A plain-language explanation for the project team')
    p.h('The project in one paragraph')
    p.p('PathSnap converts a noisy sequence of GPS locations into the road route a vehicle most likely travelled. The repository now covers reading the dataset, exploring data quality, generating nearby road candidates, computing 22 features and training classifiers. The next major stage is to connect candidate scores through the road network using HMM/Viterbi.')
    p.h('Verified state of this branch')
    p.table(['Stage','Status'],[
        ['Dataset exploration and quality record','Completed'],['Candidate generation and 22 features','Completed'],['Trip-based folds and training pipeline','Implemented and tested'],['Four baseline cross-validation runs','Completed; results archived'],['AdaBoost and final model selection','Pending'],['Final fitted deployment model','Pending'],['HMM/Viterbi route decoder','Pending'],['Shared API and two-client integration','Proposed; not verified here']],[.58,.42])
    p.h('One deployed model')
    p.p('Several models are compared during development to choose a suitable classifier. At runtime, the intended system uses one selected classifier followed by one HMM/Viterbi decoder. Two clients can call that shared backend; neither client needs all five models.')
    p.h('How to read this report')
    p.p('Pages 2–5 explain the completed data and training work. Page 6 presents measured results. Pages 7–8 explain the proposed two-client architecture and HMM/Viterbi. Pages 9–10 cover reproduction, verification and the remaining tasks.')
    p.p('Scope: this report describes the local model-training branch and recorded experiment evidence. The two-client arrangement is an architectural interpretation of the discussion, not a claim that two clients or an API have already been implemented.')
    p.end()

    p=Page(pdf,2,'Dataset and road representation','What the files mean and why they matter')
    p.table(['Partition','Trips','GPS points','Candidate rows'],[['Training','80','208,417','2,797,071'],['Student testing','10','19,160','329,247']],[.29,.12,.23,.36])
    p.h('Each trip has four file types')
    p.bullet('.track: longitude, latitude and timestamp in seconds for each GPS observation. These are the noisy measurements we want to match.')
    p.bullet('.nodes: road junction coordinates. A row index acts as the node ID inside that trip.')
    p.bullet('.arcs: directed road links. Each row contains a source node and destination node; its row index is the arc ID.')
    p.bullet('.route: an ordered list of reference road arcs used for label derivation and evaluation. It is not one label per GPS observation.')
    p.h('Road direction matters')
    p.p('Two opposite directions of the same road can be equally close to a GPS point. A useful prediction must therefore consider the direction of travel and which directed arcs can connect. Arc and node IDs belong to their trip; the same numeric ID in another trip represents a different object.')
    p.h('Training and testing roles')
    p.p('Training trips support model fitting and cross-validation. Student-testing trips support reporting after selection choices are fixed. Evaluator-testing data is not part of the supplied local training workflow and must not be used to tune the model.')
    p.h('Observed data quality')
    p.p('The recorded exploration found no missing raw values, invalid coordinates, duplicate timestamps or backward timestamps. It also found GPS jumps, long recording gaps, disconnected network components and some reference-route breaks. Clean file syntax does not mean every observation or reference is reliable.')
    p.p('Evidence: outputs/features/build_log.json, outputs/eda/dataset_inventory.csv and docs/data_quality_record.md. Raw files remain outside Git; the clone contains code and compact evidence, with instructions to rebuild the generated tables.')
    p.end()

    p=Page(pdf,3,'Preprocessing and candidate generation','Turning GPS points into manageable road choices')
    p.h('1. Use metric geometry')
    p.p('Longitude and latitude are angles. The pipeline projects each trip into a suitable UTM coordinate system before computing road distances, arc lengths and point-to-segment positions. This gives geometry in metres across trips from different regions.')
    p.h('2. Find plausible road arcs')
    p.p('For each GPS point, candidate generation searches for arcs within 75 metres and retains at most the nearest 40. This avoids scoring the entire road network for every observation while preserving multiple plausible choices near junctions and parallel streets.')
    p.h('3. Derive reference labels carefully')
    p.p('The reference route has fewer entries than the GPS track, so its row numbers cannot be paired directly with GPS rows. labels.py uses monotone dynamic programming to align observations to the reference route. A candidate gets label 1 when its arc equals that derived reference arc; other candidates get 0.')
    p.h('4. Preserve uncertainty and exclusions')
    p.p('An aligned reference farther than 50 metres is marked invalid for reference-based fitting and metrics. Trips 00000033 and 00000054 have unreliable reference coverage and are excluded from fitting and cross-validation. All 80 training trips remain in the inventory, while 78 are eligible for fitting.')
    p.h('5. Keep whole trips together')
    p.p('The pipeline creates five trip-based folds. All observations and candidates from one trip stay in the same fold. This prevents near-duplicate points from one trip appearing in both fitting and validation for that fold. Excluded trips retain fold -1.')
    p.h('A limitation to keep in every results discussion')
    p.p('The labels are derived from reference-route alignment, not independently supplied pointwise truth. Filtering valid references at 50 metres also makes candidate recall within a 75-metre search radius look unusually high. Report these conditions alongside accuracy; do not interpret the valid-reference miss rate as coverage of every raw observation.')
    p.end()

    p=Page(pdf,4,'The 22 engineered features','What the classifier sees for each GPS–road pair')
    p.h('One observation can produce several rows')
    p.p('A row represents a GPS observation paired with one candidate arc. The classifier estimates how likely that candidate is to match the derived reference. IDs, reference labels, folds and reference-derived distances do not enter the classifier feature vector.')
    groups=[
      ('Distance and competition','dist_m, dist_rank, dist_rel_min, n_cand','How close is this road, how does it rank, and how many alternatives exist?'),
      ('Position and arc geometry','frac_along, arc_len_m, to_node_dist_m, from_node_dist_m','Where is the point along the directed arc, and how close is it to either end?'),
      ('Direction and movement','heading_diff_deg, cos_heading, along_advance_m','Does road direction agree with the vehicle heading and neighbouring progress?'),
      ('Time and speed','speed_mps, dt_prev, dt_next','How fast is movement, and are sampling gaps affecting the evidence?'),
      ('Neighbouring observations','win_mean_dist, win_max_dist, win_min_dist, win_hit_frac','Do nearby observations support the same candidate geometry?'),
      ('Topology and consistency','out_deg_to, in_deg_from, has_reverse, speed_dist_resid','What road connectivity exists, and does along-arc movement agree with displacement?')]
    for title,names,meaning in groups:
        p.h(title); p.p(names,size=9.5); p.p(meaning)
    p.p('Neighbour features use up to three observations on either side, with time-gap safeguards. This is an offline trip-processing design: it can use future points and is not yet a causal live-streaming model.',size=9.5)
    p.end()

    p=Page(pdf,5,'Training and model comparison','Implemented workflow versus completed experiment runs')
    p.h('Five supported classifiers')
    p.p('The code supports Logistic Regression, Gaussian Naive Bayes, Decision Tree, Random Forest and AdaBoost. Logistic Regression has a 2,000-iteration limit. Random Forest uses 200 trees by default. AdaBoost uses 100 depth-one trees with learning rate 0.5. The shared random seed is 42.')
    p.h('Fair validation and preprocessing')
    p.p('Each classifier uses the same existing trip folds and label exclusions. Median imputation is fitted only on each fitting fold. Logistic Regression and Gaussian NB also use fold-local standard scaling. Known-reference candidate misses remain errors in observation-level Top-1 evaluation; unresolved references are excluded from reference-based denominators.')
    p.h('How the intended final model is chosen')
    p.p('The selection rule maximizes mean fold Top-1 accuracy, then uses lower binary candidate log loss and faster candidate-scoring time as tie breakers. Optuna TPE can tune the winning baseline for 30 trials by default, using the same folds. A tuned model is retained only if its CV ranking improves under the rule; student-testing does not pick the winner.')
    p.h('What actually finished')
    p.p('Five-fold cross-validation completed for Logistic Regression, Gaussian NB, Decision Tree and Random Forest. Their metrics, original logs and comparison figures are archived. Temporary fitted fold models were not retained as deployment artifacts. AdaBoost has a prepared Kaggle notebook and import workflow, but no completed AdaBoost result is claimed in this snapshot.')
    p.h('What is still required before deployment')
    p.p('Complete the remaining comparison required by the project, finalize selection and tuning, refit the chosen pipeline, save its configuration and run student-testing. selection.json is the intended pointer to the final artifact after the full training workflow completes. It is not present as a completed final selection in the archived four-model evidence.')
    p.end()

    rows=list(csv.DictReader((ROOT/'outputs/experiments/run_001/four_model_comparison.csv').open()))
    p=Page(pdf,6,'Recorded results','Classifier-only five-fold cross-validation; no decoded-route claim')
    p.table(['Model','Top-1','Fold SD','Macro F1','Log loss'],[[r['model'].replace('Gaussian Naive Bayes','Gaussian NB'),f"{float(r['top1_accuracy'])*100:.2f}%",f"{float(r['top1_accuracy_std'])*100:.2f} pp",f"{float(r['macro_f1']):.4f}",f"{float(r['log_loss']):.4f}"] for r in rows],[.34,.16,.17,.17,.16])
    p.h('What these numbers mean')
    p.bullet('Top-1: select the highest-scoring candidate at each observation. Divide correct choices by all known-reference observations, including candidate misses and observations with no candidates as errors.')
    p.bullet('Fold SD: variation in Top-1 accuracy across the five folds, in percentage points. It is not a confidence interval.')
    p.bullet('Macro F1: candidate-level binary F1 averaged across the positive and negative classes, using a fixed probability threshold of 0.5.')
    p.bullet('Log loss: candidate-level probability error. Lower values are better. A model can have acceptable Top-1 ranking while producing poor probability estimates.')
    p.h('Current interpretation')
    p.p('Random Forest has the highest recorded mean Top-1 accuracy among the four completed runs: 85.63%. Logistic Regression follows at 85.21%, a difference of about 0.42 percentage points. This does not establish a statistically significant advantage, a final five-model winner or the best HMM emission model.')
    p.p('Recorded mean fold fitting time was approximately 597 seconds for Random Forest versus 15.6 seconds for Logistic Regression. These are local classifier-pipeline timings, not API latency, whole-run duration or route-decoding time.')
    p.h('Why HMM still matters')
    p.p('These scores choose each GPS point independently. A sequence can alternate between individually plausible but disconnected roads. HMM/Viterbi is intended to add route continuity. Its effect must be measured; an accuracy improvement has not yet been demonstrated.')
    p.p('Source: outputs/experiments/run_001/four_model_comparison.csv and per-model cv_folds.csv / cv_summary.json. The archive includes six-row score samples, not full prediction tables.',size=9.5)
    p.end()

    p=Page(pdf,7,'One model, two clients','Proposed shared-backend arrangement')
    p.h('Development and deployment serve different purposes')
    p.p('During development, several models provide a comparison and report evidence. During deployment, one selected classifier provides candidate probabilities. The HMM/Viterbi decoder then combines those probabilities with road connectivity. The decoder is a sequence algorithm, not another candidate classifier in the five-model comparison.')
    p.h('Proposed request flow')
    labels=[('Client A','GPS trip request'),('Client B','GPS trip request')]
    for x,(title,detail) in zip([.09,.54],labels):
        p.ax.add_patch(FancyBboxPatch((x,.565),.36,.085,boxstyle='round,pad=.01',facecolor='#edf3f7',edgecolor='#cbd6e1'))
        p.ax.text(x+.18,.619,title,ha='center',fontsize=12,weight='bold',color=NAVY)
        p.ax.text(x+.18,.589,detail,ha='center',fontsize=9,color=INK)
        p.ax.annotate('',xy=(.495,.521),xytext=(x+.18,.558),arrowprops={'arrowstyle':'->','color':TEAL,'lw':1.6})
    p.ax.add_patch(FancyBboxPatch((.14,.34),.71,.17,boxstyle='round,pad=.015',facecolor=NAVY,edgecolor='none'))
    p.ax.text(.495,.478,'SHARED BACKEND',ha='center',fontsize=12,color='white',weight='bold')
    p.ax.text(.495,.437,'Candidates + 22 features → selected classifier',ha='center',fontsize=10,color='white')
    p.ax.text(.495,.397,'HMM/Viterbi → matched route response',ha='center',fontsize=10,color='white')
    p.y=.297
    p.h('What both clients share')
    p.p('Both clients should use the same model version, ordered feature definitions and decoder settings. Trip-specific candidates, graph data and decoding state stay separate for each request. The backend returns predicted road arcs and geometry for display; reference-route files and evaluation labels stay outside prediction responses.')
    p.h('What has not been established')
    p.p('The exact client types, API contract, server implementation and deployment setup have not been confirmed here. The diagram is a proposed interpretation of “two-client architecture.” API/UI integration is future work in this branch; no completed end-to-end service is claimed.')
    p.end()

    p=Page(pdf,8,'How HMM/Viterbi will fit','Next-stage design, explained without assuming it already exists')
    p.h('States: candidate road arcs')
    p.p('At time t, each candidate arc is a possible hidden state. The GPS point is the observed measurement. Different observations can have different candidate sets, including no candidates at all.')
    p.h('Emissions: evidence from the selected classifier')
    p.p('The classifier provides a positive-class probability for each candidate. These are independent binary scores, not a normalized distribution over roads at that observation. The decoder must define emission conversion, probability clipping and any normalization. Use log scores to avoid numerical underflow over long trips.')
    p.h('Transitions: evidence from the road network')
    p.p('A transition scores how plausible it is to move from one candidate to the next. Use directed road connectivity and metric shortest-path distances, accounting for the remaining part of the first arc and the entry position on the next arc. Compare network travel with GPS displacement and time gaps. Straight-line closeness alone cannot establish reachability.')
    p.h('Viterbi: choose the strongest whole sequence')
    p.p('For each candidate, keep the best total score of any path reaching it and remember which previous candidate produced that score. After the last observation, backtrack through these choices to recover the sequence. The chosen sequence can reject an isolated high classifier score when surrounding roads make that choice implausible.')
    p.code('best[t, j] = log_emission[t, j] + max over i of:\n             (best[t-1, i] + log_transition[i, j])\nSave the maximizing predecessor i for backtracking.')
    p.h('Edge cases that need explicit policies')
    p.p('Handle forward motion on the same arc, reverse-direction restrictions, disconnected paths, zero-length arcs, long gaps, empty candidate sets and ties. If a trip cannot connect, record a segment break rather than silently inventing a road path. Reconstruct intermediate network arcs so the displayed route is continuous where a valid connection exists.')
    p.p('Tune decoder parameters using training folds. Evaluate student-testing only after choices are fixed, comparing decoded routes with the independent classifier Top-1 baseline. Preserve original trip-local arc IDs.',size=9.5)
    p.end()

    p=Page(pdf,9,'Reproduce and inspect the work','Repository files, commands and local-only artifacts')
    p.h('Get the branch that contains the work')
    p.code('git clone --branch model-training \\\n  https://github.com/Rohansai2310/Ml_hackathon1.git\ncd Ml_hackathon1\npython3 -m venv .venv\nsource .venv/bin/activate\npython -m pip install -r requirements.txt\npython -m pip install -e .')
    p.h('Supply data and rebuild')
    p.p('Place the original supplied train/ and student_test/ folders under data/raw/. The existing local data link points to this machine and is not a portable dataset distribution. A fresh clone needs its own extraction.')
    p.code('pathsnap check\npathsnap all\npython -m pytest -q tests\npython -m pathsnap.train --feature-dir outputs/features \\\n  --output outputs/training/hmm_baseline_001 \\\n  --trials 30 --jobs 2')
    p.p('The final command starts full training and can take substantial time. Use a fresh output directory. It is not required just to read archived results or design the decoder.',size=9.5)
    p.h('Where to look')
    p.bullet('src/pathsnap/: dataset I/O, geometry, candidates, features, labels, model pipelines, scoring, training and maps.')
    p.bullet('outputs/experiments/run_001/: compact recorded evidence. It is an archive, not a resumable training run.')
    p.bullet('notebooks/four_models_walkthrough.ipynb: executed explanation of the four CV runs. Re-executing data-preview cells requires rebuilt Parquets.')
    p.bullet('notebooks/adaboost_kaggle.ipynb and docs/kaggle.md: optional AdaBoost handoff. No remote execution is claimed.')
    p.bullet('docs/guidelines/: the five supplied task PDFs. docs/HMM_VITERBI_HANDOFF.md: decoder integration starting point.')
    p.end()

    p=Page(pdf,10,'Verification and remaining work','A concrete checklist for moving into sequence decoding')
    p.h('What was verified')
    p.p('The existing project test suite passed: 26 tests. It covers the data adapter, trip/fold rules, candidate-score alignment, missing candidates, unresolved references, model serialization, synthetic training and Kaggle handoff checks. The four-model walkthrough was regenerated and verified against recorded files. These checks do not establish HMM accuracy, deployment readiness or two-client service behavior.')
    p.h('Recommended next steps')
    for text in [
        'Use one baseline classifier while implementing the decoder. A provisional model choice should be identified as provisional; deployment still needs a saved fitted pipeline.',
        'Implement the directed graph and partial-arc transition distances. Test small hand-checkable graphs before running whole trips.',
        'Implement log-space Viterbi and backtracking with deterministic ties and explicit gap/disconnection policies.',
        'Finish the remaining model experiment and final selection required by the project; export the fitted model and ordered feature configuration.',
        'Measure classifier-only versus decoded-route performance on fixed splits, recording reference limitations, candidate misses and unresolved counts.',
        'Define the shared API response and integrate both clients with the same model/decoder version. Validate request isolation and route display.'
    ]: p.bullet(text)
    p.h('Evidence and document scope')
    p.p('Implementation baseline: commit 0d1759d on model-training. This report uses the source code, build log, archived CV summaries/folds, training notes and data-quality record in that branch. It does not infer completion from a dependency list or a proposed diagram. Dates refer to 18 November 2026, Asia/Kolkata.')
    p.p('Repository: https://github.com/Rohansai2310/Ml_hackathon1\nPrimary notes: docs/training.md, docs/HMM_VITERBI_HANDOFF.md, docs/four_model_demo.md.\nRecorded results: outputs/experiments/run_001/four_model_comparison.csv.',size=9.5)
    p.end()
print(OUT)
