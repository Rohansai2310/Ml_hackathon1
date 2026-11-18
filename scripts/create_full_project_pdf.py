"""Generate the current project textbook and complete numbered application source.

Documentation-only dependency: pip install reportlab. Reads saved evidence; no fitting.
Run from any directory: python scripts/create_full_project_pdf.py
"""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
import subprocess
import textwrap
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageBreak, PageTemplate, Paragraph, Preformatted,
    Spacer, Table, TableStyle, KeepTogether,
)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.graphics.shapes import Drawing, Rect, String, Line

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SNAPSHOT = 'c74e243'
OUTPUT = ROOT / 'docs/PathSnap_Complete_Project_and_Code_Guide.pdf'
MANUSCRIPT = ROOT / 'docs/full_project_guide.md'
EVIDENCE = ROOT / 'outputs/experiments'
NAVY, TEAL, GREY = '#15324d', '#087e86', '#566679'
WIDTH, HEIGHT = A4
FONT_DIR = Path('/usr/share/fonts/truetype/dejavu')
for name, filename in [('Body', 'DejaVuSans.ttf'), ('BodyBold', 'DejaVuSans-Bold.ttf'),
                       ('Code', 'DejaVuSansMono.ttf')]:
    pdfmetrics.registerFont(TTFont(name, str(FONT_DIR / filename)))
pdfmetrics.registerFontFamily('Body', normal='Body', bold='BodyBold', italic='Body', boldItalic='BodyBold')

STYLES = getSampleStyleSheet()
STYLES.add(ParagraphStyle('Text', fontName='Body', fontSize=9.1, leading=14.2, spaceAfter=9, textColor=colors.HexColor(NAVY)))
STYLES.add(ParagraphStyle('Chapter', fontName='BodyBold', fontSize=19, leading=25, spaceAfter=17, textColor=colors.HexColor(NAVY)))
STYLES.add(ParagraphStyle('Sub', fontName='BodyBold', fontSize=10.5, leading=15, spaceBefore=10, spaceAfter=7, textColor=colors.HexColor(TEAL)))
STYLES.add(ParagraphStyle('Note', fontName='Body', fontSize=7.5, leading=10.7, spaceAfter=4, textColor=colors.HexColor(GREY)))
STYLES.add(ParagraphStyle('Mono', fontName='Code', fontSize=6.7, leading=9.3, backColor=colors.HexColor('#f0f4f8'), borderPadding=7, spaceAfter=9))
STYLES.add(ParagraphStyle('CoverTitle', fontName='BodyBold', fontSize=32, leading=40, textColor=colors.HexColor(NAVY), spaceAfter=22))
STYLES.add(ParagraphStyle('CoverSub', parent=STYLES['Text'], fontSize=14, leading=22, textColor=colors.HexColor(TEAL), spaceAfter=20))

MODULES = {
    'config': 'Authoritative constants, paths and the ordered 22-feature schema. Thresholds affect both training and inference; changing them requires new validation.',
    'data_io': 'Raw file reader and Trip structure. The with_route flag separates label-bearing reads from inference reads.',
    'geo': 'Projection and vectorized point-to-segment geometry in metres; heading and angular differences.',
    'network': 'Builds arrays indexed by original directed arc IDs and node IDs for one trip.',
    'candidates': 'Uses a cropped spatial index to find nearby directed arcs, while preserving trip-local identities.',
    'features': 'Creates all 22 label-free candidate features from local geometry, time, direction, motion and topology.',
    'labels': 'Reference-only monotone alignment that derives pointwise labels. This is not the inference decoder.',
    'eda': 'Exploration and quality evidence on train/student_test, including reference coverage, sampling, radius studies and plots.',
    'build': 'Creates candidate/point Parquets and preserves whole-trip folds and label validity.',
    'data': 'Validates generic model tables and complete observation inventories; distinguishes unresolved labels from negatives.',
    'training_handoff': 'Converts native feature Parquets into validated model tables without changing folds or exclusions.',
    'dataset': 'Convenience native-table loaders for teammates. Filtering is explicit and consistent with eligibility rules.',
    'models': 'Builds five optional classifier pipelines and their Bayesian search spaces. Runtime uses the selected forest only.',
    'metrics': 'Binary candidate quality and pointwise road-choice metrics with explicit reference/candidate-miss accounting.',
    'train': 'General comparison/CV/optional Bayesian-optimization runner. Its implementation does not prove every optional experiment was completed.',
    'selected_forest': 'The actual fixed Random Forest refit path: 200 trees, seed 42, eligible training labels only, frozen artifact and student scores.',
    'hmm': 'Reviewed directed metric HMM-style decoder, bounded Dijkstra, log-space Viterbi, block backtracking and route reconstruction.',
    'tune_hmm': 'Selects decoder settings on original training OOF scores; freezes the configuration before student validation.',
    'evaluate_forest': 'Predicts every evaluator trip before reading any reference route, then computes metrics on frozen predictions.',
    'api': 'Shared FastAPI backend loading the frozen classifier and optional validated decoder settings for both clients.',
    'maps': 'Folium exploration/display helper. For segmented decoder output, call/display segments separately rather than connecting breaks.',
    'cli': 'Dataset checks, exploration, feature building, inspection and example map commands. Modeling modules have separate CLIs.',
    'kaggle_bundle': 'Optional AdaBoost handoff packaging/import; imported artifacts must meet the local schema and provenance checks.',
    'report_baselines': 'Recreates the four recorded baseline comparison and diagnoses tied classifier scores.',
    'docgen': 'Writes the historical exploration-stage data-quality record. Some hardcoded stage wording predates evaluator integration.',
    '__init__': 'Package marker/documentation; it does not train or serve a model.',
    '__main__': 'Allows python -m pathsnap to dispatch to the dataset command-line interface.',
}

PURPOSE = {
    'generate_candidates': 'Crop the network, query nearby line geometries, calculate exact distances and cap the nearest candidates per point.',
    '_neighbour_index': 'Return valid neighbour indices plus a mask; missing or distant neighbours refer back to self and are masked out.',
    'point_motion': 'Estimate speed and compass heading from permitted adjacent observations; provide time gaps and neighbouring displacement.',
    'build_candidate_table': 'Create one candidate feature row per point/arc pair. Window context uses future points, so inference is offline.',
    's_along': 'Project neighbouring point displacement onto the candidate arc direction without clamping to endpoints.',
    'align_points_to_route': 'Use prefix-minimum dynamic programming to align GPS points monotonically to reference-route positions, then backtrack.',
    'process_trip': 'Build candidates and the full point inventory; attach reference labels only when explicitly requested.',
    'make_folds': 'Keep each eligible trip in one validation fold; native provided assignments are retained by the training runner.',
    'convert': 'Translate IDs, blank unresolved labels, validate inventory consistency and preserve original trip folds.',
    'validate': 'Reject malformed identities, duplicate rows, reference mismatches, invalid labels and forbidden feature columns.',
    'build_model': 'Construct fold-local imputation and the requested estimator; scale only the models that require it here.',
    'score_candidates': 'Preserve candidate IDs and extract the fitted positive-class probability for every candidate row.',
    'evaluate': 'Join the selected candidate back to every observation, count known-reference misses and compute candidate F1/log loss.',
    'cross_validate': 'Fit a new pipeline for each fold and save candidate probabilities produced without fitting that held-out trip.',
    'fit_selected': 'Fit the user-selected baseline forest on valid eligible training rows, save and hash it, then score student testing.',
    'file_hash': 'Stream the file into SHA-256 in 1 MB chunks so large artifacts need not be copied into memory.',
    'write_json': 'Serialize provenance and configuration while converting nonfinite sentinel floats into strings for valid JSON.',
    'load_decoder_parameters': 'Require the exact classifier hash, evaluator-free selection record and allowed finite decoder parameters.',
    'RoadGraph': 'Directed adjacency and metric arc lengths; parallel arc IDs remain distinct, with a bounded local-search cache.',
    'search': 'Dijkstra search within a rounded metric limit; reuse/move cached entries and evict the oldest after 512 entries.',
    'transition': 'Return partial-arc plus directed shortest-path distance and intermediate arcs; small same-arc backward jitter is a hold.',
    'viterbi_decode': 'Validate aligned inputs, create log emissions, maximize connected sequence scores, handle breaks and reconstruct routes.',
    'finish': 'Backtrack the current connected block independently, assign original point IDs and segment IDs, then reset block state.',
    'limit_between': 'Bound travel by elapsed time and maximum speed plus candidate projection allowance, capped at maximum distance.',
    'load_trips': 'Align frozen candidate scores one-to-one with geometry and point labels; check original training-fold provenance.',
    'evaluate_trips': 'Run independent ranking or decoder, check directed reconstruction and aggregate fold/trip/pooled accuracy.',
    'settings_grid': 'Return the historical baseline plus 24 beta/jitter/emission combinations, totalling 25 decoder configurations.',
    'screen_subset': 'Deterministically sample up to two training trips per original fold, without replacement, using seed 42.',
    'prepare_trip': 'Read no reference route, validate track/network, build features and match float32 training precision.',
    'reference_partition': 'After predictions freeze, derive reference labels and convert them into the metric inventory.',
    'pointwise_matches': 'Compare decoded arc choices with valid derived references, using the same known-point denominator as RF.',
    'nearest_scores': 'Construct the distance-only baseline using a monotone exponential transformation of candidate distance.',
    'load_model': 'Validate complete metadata, ordered features and file hash before loading the trusted local joblib artifact.',
    'set_model': 'Register a classifier exposing predict_proba; normal server startup uses validated load_model.',
    'match_trip': 'Serve the complete label-free pipeline for one stored trip; return point matches, route segments and explicit breaks.',
    'health': 'Report that the API responds and whether model/selected decoder configuration were loaded.',
    'MatchRequest': 'Pydantic request schema: an eight-digit trip ID, a partition and optional server data-root override.',
    'MatchResponse': 'Response schema containing per-point arc IDs, route segments, break reasons and matching counts.',
    'DecodeResult': 'Decoder output container; -1 preserves unmatched observations rather than dropping them.',
    'DecoderTrip': 'In-memory bundle of one trip geometry, candidates, frozen scores, original fold and derived evaluation labels.',
    'FoldPlan': 'Materialize only the current fold arrays, reducing memory use for millions of candidate rows.',
    'Trip': 'Raw trip data container; the reference route is optional and absent on the inference path.',
    'TripGeom': 'Metric geometry bundle, retaining a direct index relationship to original trip nodes/arcs.',
    'Partition': 'Validated candidate table, complete observation inventory, ordered feature list and optional trip folds.',
}

ASSIGNMENTS = {
    'progress': 'signed projected travel along the previous arc, in metres',
    'same_hold': 'mask for same-arc motion within the permitted backwards projection tolerance',
    'totals': 'every predecessor score minus its network/GPS distance disagreement penalty',
    'back': 'predecessor indices used to reconstruct the selected sequence (or the reference DP backpointer table)',
    'next_scores': 'best incoming transition score plus the current candidate log emission',
    'distance': 'the metric distance for the current path or geometry calculation',
    'distances': 'reachable-node distances or the matrix of candidate-pair travel distances in this scope',
    'emissions': 'current candidates’ log emission potentials',
    'point_arcs': 'complete per-observation predicted arc array initialized with unmatched value -1',
    'segment_ids': 'per-observation connected-block identity, preserving unmatched positions',
    'dp': 'the current row of best reference-alignment costs',
    'pm': 'the running prefix minimum allowing only nondecreasing reference-route positions',
    'valid': 'validity mask or count for the current reference/neighbour computation',
    'known': 'the observations with valid derived reference labels',
    'correct': 'known-reference predictions that equal the aligned reference arc',
    'positive_indices': 'the column index for binary class 1 in the classifier output',
    'positive_probabilities': 'the Random Forest scores passed to HMM emissions',
    'artifact_hash': 'the classifier artifact fingerprint, binding the selected decoder to its model',
    'model_hash': 'the frozen classifier file fingerprint used throughout evaluation',
    'finalists': 'the original decoder plus the three strongest screening trials, without duplicates',
    'best': 'the training-selected candidate with fold score, trip score and trial-order tie breaking',
    'selected': 'the frozen decoder settings plus model identity and training selection evidence',
    'fit': 'eligible labelled training rows (or the current fold training table in cross-validation)',
    'frac': 'fractional projection locations of current candidates along their directed arcs',
    'chord': 'neighbouring GPS displacement used as a direction-consistency reference',
    'adv': 'signed neighbouring progress along each candidate’s directed axis',
    'resid': 'absolute along-arc progress minus neighbouring GPS displacement',
    'rank': 'zero-based candidate distance rank within each observation',
    'whit': 'fraction of valid context neighbours close to the candidate arc',
    'wmean': 'mean valid neighbour distance, falling back to current-point distance',
    'wmin': 'minimum valid neighbour distance, falling back to current-point distance',
}

# Domain-specific notes override the mechanical statement descriptions at the
# exact frozen source positions. All other statements still receive syntax notes.
LINE_NOTES = {
    'hmm': {
        87: 'Measure signed movement between the two projections on the previous arc. A negative value means the projected observation moved backwards.',
        88: 'Use direct same-arc travel when forward or within the selected backwards jitter allowance. Otherwise search for a legal directed return path.',
        91: 'Clamp permitted small backwards jitter to zero metres: stay on the same road rather than drive against its direction.',
        93: 'Start an intermediate network search at the destination node of the previous directed arc.',
        94: 'End that intermediate search at the source node of the current directed arc.',
        95: 'Retrieve shortest distances and predecessor arc IDs from the previous arc endpoint, within the travel bound.',
        96: 'Add remaining previous-arc metres, shortest intermediate path metres, and current-arc metres up to its projection.',
        100: 'Prepare to reconstruct the intermediate path backwards from the target node.',
        102: 'Look up the actual incoming arc chosen by Dijkstra for this node; this preserves parallel-arc identity.',
        104: 'Move to that arc’s source node and continue backtracking until the original source is reached.',
        105: 'Return the total metric distance and reverse the backwards-collected intermediate arcs into travel order.',
        149: 'Clip zero classifier scores before taking logs. A probability of one contributes zero; smaller probabilities contribute negative evidence.',
        153: 'Group candidates by their original GPS observation index, sorting numeric arc IDs so exact decoder ties are deterministic.',
        162: 'Choose the highest-scoring final candidate in this connected block as the start of backtracking.',
        165: 'Record the chosen original point index, directed arc ID and fraction, keeping geometry needed for route reconstruction.',
        167: 'Follow the stored predecessor for this candidate into the preceding observation. No artificial predecessor is inserted.',
        168: 'Reverse the collected matches so they follow chronological GPS order.',
        171: 'Write choices into complete-trip arrays using original observation indices, leaving unmatched positions at -1.',
        173: 'Reset only the just-completed block’s state. Its final choices have already been saved.',
        177: 'Limit travel using elapsed time times maximum speed plus two candidate radii; apply the overall distance cap.',
        183: 'Visit every GPS observation, including indices absent from the candidate table. This preserves missing-point identities.',
        196: 'Initialize a new independent block using the current emissions; no transition from a previous block is invented.',
        200: 'Measure straight-line GPS displacement in projected metres, for comparison with network travel distance.',
        201: 'Allocate one travel distance for every previous/current candidate pair, initially marking all pairs unreachable.',
        206: 'Read the previous candidate arc’s destination node, from which legal forward network travel begins.',
        207: 'Run or reuse a bounded shortest-path search for this previous candidate endpoint.',
        208: 'Look up distances to each current candidate’s source node; nodes not reached remain infinitely far away.',
        209: 'Build all candidate-pair network distances from partial previous arc + intermediate path + partial current arc.',
        213: 'For permitted same-arc holds/forward motion, replace the endpoint-route formula with direct projected progress clamped at zero.',
        214: 'Reject candidate movements exceeding the exact physical travel limit, even if the rounded cached search reached them.',
        215: 'Construct all predecessor-to-current scores: previous best score minus absolute network/GPS disagreement divided by beta.',
        216: 'For each current candidate (axis zero), save the previous candidate giving its greatest transition-adjusted score.',
        217: 'Gather those best predecessor scores and add each current candidate’s log emission. This is the Viterbi recurrence.',
        218: 'If every current score is unreachable, finish the old block independently and restart at the current point.',
        225: 'Subtract a common maximum from all reachable scores. Relative path rankings and backtracking choices remain unchanged.',
        227: 'Store the predecessor vector so finish() can recover the globally best path through this block.',
        235: 'Recalculate the chosen transition with the same distance/jitter rules to obtain the actual intermediate road arc IDs.',
        240: 'A same-arc forward movement or tolerated hold needs no additional copy of that road in the route.',
        243: 'Insert unobserved intermediate arcs so consecutive output route arcs share directed endpoints.',
        246: 'Package the complete observation matches, block identities, separate reconstructed routes and explicit break causes.',
    },
    'features': {
        32: 'A neighbour is usable only when its timestamp is within 10 seconds times its offset; boundary/gap neighbours are excluded.',
        42: 'Compute central neighbouring displacement, using self as the safe fallback where a neighbour is invalid.',
        44: 'Speed equals the displacement magnitude divided by the available time difference; use zero when no positive interval exists.',
        46: 'Use direction evidence only for movement at least 1 m/s over a positive interval.',
        65: 'Normalize each arc direction vector using its metric length; protect zero-length arcs with a safe denominator.',
        68: 'Find the first candidate row for each point; subtracting it from row position gives within-point distance rank.',
        70: 'Count candidates per observation, including zero counts for observations that have no candidate row.',
        71: 'Reduce sorted candidate distances within each nonempty observation to obtain its nearest-road distance.',
        73: 'Compute Euclidean GPS distance to each candidate’s destination node, a position/junction feature.',
        74: 'Compute distance to each candidate’s source node, preserving the arc’s direction-dependent endpoint identity.',
        77: 'Compare estimated travel heading with the candidate directed-arc bearing; reverse arcs generally disagree by about 180 degrees.',
        79: 'Set heading difference to a neutral 90 degrees when heading is unreliable, rather than inventing a travel direction.',
        83: 'Take a dot product with the directed unit arc vector to measure unclamped signed progress along that road.',
        87: 'Subtract preceding from following projection to get signed neighbouring advance on this candidate road.',
        93: 'Evaluate offsets -3, -2, -1, +1, +2 and +3; the current point is excluded from neighbour summaries.',
        95: 'For this offset, measure the neighbour GPS point’s distance to the same candidate arc, not to its own nearest road.',
        99: 'Count valid neighbours within 37.5 metres of the arc and count all usable neighbours for the hit-fraction denominator.',
        105: 'Assemble identifiers and all 22 feature arrays, broadcasting per-point motion values to that point’s candidate rows.',
    },
    'labels': {
        26: 'Allocate N by M reference distances: one row per GPS point, one column per position in the ordered reference route.',
        28: 'Measure every GPS point against this reference arc; repeat endpoints so point_segment handles the whole column at once.',
        30: 'Keep uncapped distances for later validity checks. The 150-metre alignment cap must not hide invalid references.',
        31: 'Cap alignment costs at 150 metres in place, limiting the effect of very distant GPS points on reference order.',
        33: 'Initialize alignment costs for the first GPS point; any reference-route starting position is allowed.',
        36: 'For each current reference position j, find the cheapest previous cost among positions k <= j. This enforces monotone order.',
        37: 'Record an index attaining each prefix minimum; the accumulated indices make predecessor recovery deterministic.',
        39: 'Add current distance cost to the cheapest allowed preceding alignment cost for every route position.',
        40: 'Choose the cheapest final route position, then backtrack through the saved predecessor table.',
        45: 'Translate ordered route positions into actual trip-local arc IDs; a revisited arc can occupy different route positions.',
        46: 'Return route positions, derived arc labels and original uncapped distances used for the 50-metre validity rule.',
    },
    'selected_forest': {
        23: 'Load only the training and student-testing feature tables and verify their fixed schema, folds and disjoint trip identities.',
        28: 'Record eligible/excluded trips, input fingerprints, software versions and the fact evaluator data did not choose this classifier.',
        42: 'Construct the selected baseline pipeline with 200 Random Forest trees and median imputation; this does not fit it yet.',
        47: 'Fit imputation and the forest on all eligible training candidate labels. No student/evaluator labels enter this call.',
        51: 'Save a compressed model to a pending artifact so an incomplete write is not presented as the final model filename.',
        52: 'Promote the finished pending file to pipeline.joblib, the artifact later loaded by evaluation and the API.',
        55: 'After fitting is finished, score student-test candidates for post-fit reporting and later frozen decoder validation.',
        60: 'Verify original and reloaded pipelines produce the same scores on up to 1,000 student rows within absolute tolerance 1e-12.',
        66: 'Mark the run complete only after model serialization and student scoring/reload checks succeed.',
    },
    'api': {
        112: 'Load stored trip GPS and road data with with_route=False. The request prediction cannot use reference labels.',
        125: 'Cast the same continuous columns to float32 as the training Parquets, avoiding training/serving precision mismatch.',
        133: 'Select exactly the configured 22 columns in their fixed model input order; identifiers and references are excluded.',
        135: 'Run the single shared fitted classifier over all candidate rows for this request.',
        141: 'Locate binary class 1 explicitly using the fitted estimator’s class labels, rather than assuming a probability-column index.',
        145: 'Extract one positive-class compatibility score for every candidate while retaining candidate row alignment.',
        147: 'Decode scores using metric road geometry, complete observation times and the validated selected settings from startup.',
        158: 'Return matches and separate connected routes for the requested trip. No reference route or reference accuracy is included.',
    },
}


def para(text, style='Text'):
    return Paragraph(escape(str(text)), STYLES[style])


def snapshot_source(relative):
    """Keep the historical numbered guide reproducible after application changes."""
    return subprocess.check_output(['git', 'show', f'{SOURCE_SNAPSHOT}:{relative}'], cwd=ROOT).decode()


def short(node, limit=150):
    value = ast.unparse(node).replace('\n', ' ')
    return value if len(value) <= limit else value[:limit - 3] + '...'


def call_meaning(node):
    name = short(node.func, 80)
    special = {
        'model.fit': 'fit the pipeline using these training features and binary candidate labels',
        'model.predict_proba': 'produce one probability per fitted class for each feature row',
        '_model.predict_proba': 'score request candidates with the shared fitted classifier',
        'viterbi_decode': 'decode the highest-scoring feasible sequence and its connected route segments',
        'finish': 'backtrack and close the current block before continuing',
        'write_json': 'save structured configuration, metrics or provenance to the specified path',
        'file_hash': 'compute SHA-256 of the referenced file',
        'joblib.dump': 'save the trained Python pipeline artifact',
        'joblib.load': 'restore the trusted serialized pipeline',
        'build_geom': 'project raw trip coordinates and derive metric directed-arc geometry',
        'build_candidate_table': 'generate candidates and compute the same 22 label-free features',
        'align_points_to_route': 'derive point references using monotone alignment to the supplied route',
        'score_candidates': 'score rows while preserving exact candidate identity/order',
    }
    if name in special:
        return special[name]
    suffix = name.split('.')[-1]
    actions = {
        'to_csv': 'write this table to a CSV evidence/prediction file',
        'to_parquet': 'write this table in the typed Parquet format',
        'mkdir': 'create the required output directory',
        'merge': 'join tables using the supplied identity keys and validation rule',
        'sort_values': 'order rows deterministically by the specified keys',
        'groupby': 'group rows by the supplied observation, trip or fold keys',
        'append': 'add the item to the current ordered collection',
        'extend': 'append all items from the supplied sequence in order',
        'astype': 'convert the values to the specified numeric or identifier dtype',
        'read_parquet': 'load a generated Parquet table (or selected columns)',
        'read_csv': 'load a saved CSV table with the supplied identifier types',
        'heappush': 'add a candidate node distance to the Dijkstra priority queue',
        'heappop': 'take the smallest queued distance before expanding its outgoing arcs',
        'argmax': 'choose the first index with the greatest score',
        'argmin': 'choose the first index with the smallest cost',
        'clip': 'bound numeric values to the specified interval',
        'isfinite': 'check that values are neither NaN nor infinite',
        'where': 'select values according to the supplied boolean condition',
        'full': 'allocate an array initialized with the supplied fill value',
        'zeros': 'allocate a zero-initialized array of the specified shape',
    }
    return actions.get(suffix, 'call ' + name + ' with the shown arguments')


def explain(node):
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return 'Import the shown dependency or shared project definitions; no model is fitted by this statement.'
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        doc = ast.get_docstring(node)
        return ('Define ' + node.name + '. ' + PURPOSE.get(node.name, (doc.split('\n')[0] if doc else 'The following indented statements define its behavior.')))
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        destination = ', '.join(short(t, 90) for t in targets)
        if destination in ASSIGNMENTS:
            return 'Set ' + destination + ': ' + ASSIGNMENTS[destination] + '.'
        value = node.value
        if value is None:
            return 'Declare the field ' + destination + ' with the shown type.'
        if isinstance(value, ast.Call):
            return 'Set ' + destination + ': ' + call_meaning(value) + '.'
        if isinstance(value, (ast.List, ast.Tuple, ast.Dict, ast.Set)):
            return 'Set ' + destination + ' to the listed ordered values, fields or collected results.'
        return 'Set ' + destination + ' to ' + short(value) + '.'
    if isinstance(node, ast.If):
        return 'Run the following branch when ' + short(node.test) + '. Any else branch handles the complementary case.'
    if isinstance(node, (ast.For, ast.AsyncFor)):
        return 'For each ' + short(node.target) + ' in ' + short(node.iter) + ', execute this indented block.'
    if isinstance(node, ast.While):
        return 'Repeat this block while ' + short(node.test) + '; the body updates the search/backtracking state.'
    if isinstance(node, ast.Return):
        return 'Return ' + (short(node.value) if node.value else 'to the caller') + ' from this function.'
    if isinstance(node, ast.Raise):
        return 'Stop this operation with ' + (short(node.exc) if node.exc else 'the original exception') + '.'
    if isinstance(node, ast.Expr):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None  # Literal documentation is already included verbatim in the code.
        return (call_meaning(node.value) if isinstance(node.value, ast.Call) else short(node.value)) + '.'
    if isinstance(node, ast.AugAssign):
        return 'Update ' + short(node.target) + ' using ' + type(node.op).__name__ + ' with ' + short(node.value) + '.'
    if isinstance(node, ast.With):
        return 'Open/manage the shown resource for this block and close it automatically afterward.'
    if isinstance(node, ast.Try):
        return 'Attempt this operation; the following except branches translate or propagate errors.'
    if isinstance(node, ast.Continue):
        return 'Skip the rest of this iteration and continue with the next item.'
    if isinstance(node, ast.Break):
        return 'Exit this loop immediately.'
    if isinstance(node, ast.Global):
        return 'Use module-level shared state for ' + ', '.join(node.names) + '.'
    if isinstance(node, ast.Nonlocal):
        return 'Update the enclosing decoder block variables: ' + ', '.join(node.names) + '.'
    if isinstance(node, ast.Delete):
        return 'Release references to the shown objects before the next stage.'
    return 'Execute this ' + type(node).__name__ + ' statement in the surrounding block.'


def statement_end(node):
    # Compound statement notes refer to the header, not their entire nested body.
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.If, ast.For, ast.While, ast.With, ast.Try)):
        first = node.body[0].lineno if node.body else node.lineno + 1
        return max(node.lineno, first - 1)
    return node.end_lineno


class GuideDoc(BaseDocTemplate):
    def __init__(self, filename):
        super().__init__(str(filename), pagesize=A4, leftMargin=43, rightMargin=43,
                         topMargin=55, bottomMargin=46, title='PathSnap: Complete Project and Code Guide',
                         author='PathSnap project documentation', allowSplitting=True)
        self.addPageTemplates(PageTemplate(id='Guide', frames=[Frame(43, 46, WIDTH-86, HEIGHT-101,
                               leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)], onPage=self.decorate))

    def decorate(self, canvas, doc):
        canvas.saveState()
        canvas.setFillColor(colors.HexColor(NAVY))
        canvas.rect(0, HEIGHT-32, WIDTH, 32, fill=1, stroke=0)
        canvas.setFont('BodyBold', 8)
        canvas.setFillColor(colors.white)
        canvas.drawString(43, HEIGHT-21, 'PATHSNAP  |  COMPLETE PROJECT AND CODE GUIDE')
        canvas.setFont('Body', 7)
        canvas.setFillColor(colors.HexColor(GREY))
        canvas.drawString(43, 26, '18 November 2026  |  source snapshot c74e243  |  model-training')
        canvas.drawRightString(WIDTH-43, 26, str(doc.page))
        canvas.restoreState()

    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and flowable.style.name == 'Chapter':
            title = flowable.getPlainText()
            key = hashlib.sha1(title.encode()).hexdigest()
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(title, key, level=0, closed=False)
            if title != 'Contents':
                self.notify('TOCEntry', (0, title, self.page, key))


def table(headers, rows, widths=None):
    content = [[para(str(c), 'Note') for c in row] for row in [headers] + rows]
    result = Table(content, colWidths=widths, repeatRows=1, hAlign='LEFT')
    result.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#dceaf0')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#f3f6f8'), colors.white]),
        ('VALIGN', (0,0), (-1,-1), 'TOP'), ('LEFTPADDING', (0,0), (-1,-1), 7),
        ('RIGHTPADDING', (0,0), (-1,-1), 7), ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LINEBELOW', (0,0), (-1,0), .6, colors.HexColor('#aac6d3')),
    ]))
    return result


def numbered_lines(lines, start):
    printed = []
    for i, line in enumerate(lines, start):
        # Preserve indentation and original physical line numbers; wrap only for paper.
        pieces = textwrap.wrap(line.expandtabs(4), width=110, replace_whitespace=False,
                              drop_whitespace=False, break_long_words=True, break_on_hyphens=False) or ['']
        printed.append(f'{i:4d} | ' + pieces[0])
        printed.extend('     |     ' + piece for piece in pieces[1:])
    return '\n'.join(printed)


def markdown(story):
    content = MANUSCRIPT.read_text().splitlines()
    paragraph, code = [], None
    def flush():
        if paragraph:
            story.append(para(' '.join(paragraph)))
            paragraph.clear()
    for line in content:
        if line.startswith('```'):
            flush()
            if code is None:
                code = []
            else:
                wrapped = []
                for c in code:
                    wrapped.extend(textwrap.wrap(c, width=111, replace_whitespace=False,
                                   drop_whitespace=False, subsequent_indent='    ') or [''])
                story.append(Preformatted('\n'.join(wrapped), STYLES['Mono']))
                code = None
        elif code is not None:
            code.append(line)
        elif line.startswith('## '):
            flush()
            story.extend([PageBreak(), para(line[3:], 'Chapter')])
            if line.startswith('## 6.'):
                configuration = ast.parse(snapshot_source('src/pathsnap/config.py'))
                feature_columns = ast.literal_eval(next(n.value for n in configuration.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='FEATURE_COLUMNS' for t in n.targets)))
                definitions = ast.parse(snapshot_source('src/pathsnap/training_handoff.py'))
                assign = next(n for n in definitions.body if isinstance(n, ast.Assign) and any(isinstance(t,ast.Name) and t.id=='FEATURE_DEFINITIONS' for t in n.targets))
                descriptions = ast.literal_eval(assign.value.args[0].args[1])
                story.append(table(['#', 'Feature', 'Meaning'], [[str(i),n,d] for i,(n,d) in enumerate(zip(feature_columns, descriptions),1)], [24,155,WIDTH-265]))
                story.append(Spacer(1,12))
            if line.startswith('## 14.'):
                add_results(story)
            if line.startswith('## 1.'):
                story.extend([pipeline_diagram(),Spacer(1,14)])
            if line.startswith('## 16.'):
                story.extend([client_diagram(),Spacer(1,14)])
        elif line.startswith('# '):
            continue
        elif not line.strip():
            flush()
        else:
            paragraph.append(line.strip())
    flush()


def read_json(path):
    return json.loads(path.read_text())


def box(drawing, x, y, width, height, label, fill=TEAL):
    drawing.add(Rect(x,y,width,height,rx=5,ry=5,fillColor=colors.HexColor(fill),strokeColor=None))
    drawing.add(String(x+width/2,y+height/2-3,label,textAnchor='middle',fontName='BodyBold',fontSize=8,fillColor=colors.white))


def arrow(drawing, x1, y1, x2, y2):
    drawing.add(Line(x1,y1,x2,y2,strokeColor=colors.HexColor(TEAL),strokeWidth=1.4))
    # Arrowheads for the vertical/downward arrows used by these diagrams.
    drawing.add(Line(x2-3,y2+5,x2,y2,strokeColor=colors.HexColor(TEAL)))
    drawing.add(Line(x2+3,y2+5,x2,y2,strokeColor=colors.HexColor(TEAL)))


def pipeline_diagram():
    drawing=Drawing(WIDTH-86,218)
    labels=['GPS + directed road network','Metric geometry + candidate arcs','22 features + one Random Forest','HMM/Viterbi + directed shortest paths','Matched points + separate route segments']
    for i,label in enumerate(labels):
        y=178-i*42
        box(drawing,72,y,360,29,label,NAVY if i in [0,4] else TEAL)
        if i<4:
            arrow(drawing,252,y-2,252,y-11)
    return drawing


def client_diagram():
    drawing=Drawing(WIDTH-86,205)
    box(drawing,30,166,175,29,'Client A (to build)')
    box(drawing,300,166,175,29,'Client B (to build)')
    arrow(drawing,117,164,175,125)
    arrow(drawing,387,164,330,125)
    box(drawing,65,91,375,32,'Shared FastAPI: one RF + selected HMM',NAVY)
    arrow(drawing,252,89,252,65)
    box(drawing,65,30,375,32,'Trip matches, segment boundaries and arc IDs')
    drawing.add(String(252,9,'Prediction reads GPS + road data; reference routes stay outside inference.',textAnchor='middle',fontName='Body',fontSize=7.2,fillColor=colors.HexColor(GREY)))
    return drawing


def add_results(story):
    current = read_json(EVIDENCE/'evaluator_test_rf_hmm_tuned_001/metrics.json')
    old = read_json(EVIDENCE/'evaluator_test_rf_hmm_001/metrics.json')
    student = read_json(EVIDENCE/'hmm_tuning_001/student_validation.json')
    rows = [
        ['Nearest candidate', f"{current['nearest_top1_accuracy']:.2%}", 'Distance baseline'],
        ['Random Forest', f"{current['top1_accuracy']:.2%}", f"{current['correct_known_observations']:,} / {current['known_observations']:,}"],
        ['RF + original HMM', f"{old['hmm_top1_accuracy']:.2%}", f"{old['hmm_correct_known_observations']:,} / {old['known_observations']:,}"],
        ['RF + selected HMM', f"{current['hmm_top1_accuracy']:.2%}", f"{current['hmm_correct_known_observations']:,} / {current['known_observations']:,}"],
        ['Student: RF + selected HMM', f"{student['selected_hmm']['pooled_top1']:.2%}", f"{student['selected_hmm']['correct']:,} / {student['selected_hmm']['known']:,}"],
    ]
    story.append(table(['Pipeline / partition', 'Pointwise Top-1', 'Correct / valid references'], rows, [200,100,WIDTH-386]))
    story.append(Spacer(1,12))
    drawing = Drawing(WIDTH-86, 142)
    for i,(label,value) in enumerate([('Nearest',current['nearest_top1_accuracy']),('RF',current['top1_accuracy']),('Original HMM',old['hmm_top1_accuracy']),('Selected HMM',current['hmm_top1_accuracy'])]):
        y=110-i*28
        drawing.add(String(0,y+3,label,fontName='Body',fontSize=8,fillColor=colors.HexColor(NAVY)))
        drawing.add(Rect(102,y,320*value,16,fillColor=colors.HexColor(TEAL if i==3 else '#8ba8bf'),strokeColor=None))
        drawing.add(String(110+320*value,y+3,f'{value:.2%}',fontName='BodyBold',fontSize=8))
    story.extend([drawing, para('Repeat evaluator benchmark; every bar uses the same valid-reference denominator.', 'Note')])
    with (EVIDENCE/'evaluator_test_rf_hmm_tuned_001/per_trip_metrics.csv').open() as stream:
        trips=list(csv.DictReader(stream))
    story.append(para('Every evaluator trip', 'Sub'))
    story.append(table(['Trip', 'Known', 'Unresolved', 'RF', 'Selected HMM'],
                       [[r['trip_id'],r['known_observations'],r['unresolved_observations'],f"{float(r['top1_accuracy']):.2%}",f"{float(r['hmm_top1_accuracy']):.2%}"] for r in trips], [105,75,85,110,WIDTH-461]))
    story.append(Spacer(1,12))


def append_source(story):
    order = list(MODULES)
    source_paths = subprocess.check_output(['git','ls-tree','-r','--name-only',SOURCE_SNAPSHOT,'src/pathsnap'],cwd=ROOT,text=True).splitlines()
    actual = {Path(p).stem for p in source_paths if p.endswith('.py')}
    if set(order) != actual:
        raise ValueError('Update module coverage before generating: '+str(actual ^ set(order)))
    manifest = []
    for index,name in enumerate(order,1):
        path = ROOT/'src/pathsnap'/f'{name}.py'
        source = snapshot_source(str(path.relative_to(ROOT)))
        lines = source.splitlines()
        tree = ast.parse(source)
        manifest.append({'file':str(path.relative_to(ROOT)), 'lines':len(lines),
                         'sha256':hashlib.sha256(source.encode()).hexdigest()})
        story.extend([PageBreak(), para(f'Code {index:02d}. {name}.py', 'Chapter'), para(MODULES[name]),
                      para(f'Exact source: src/pathsnap/{name}.py | {len(lines)} physical lines', 'Note')])
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                desc = PURPOSE.get(node.name, ast.get_docstring(node) or 'See the numbered definition and statements below.')
                story.append(para(f'L{node.lineno}: {node.name} — '+desc.split('\n')[0], 'Note'))
        nodes = sorted([n for n in ast.walk(tree) if isinstance(n,ast.stmt)],key=lambda n:(n.lineno,n.col_offset))
        for start in range(1,len(lines)+1,18):
            end = min(start+17,len(lines))
            heading=para(f'{name}.py · lines {start}–{end}', 'Sub')
            listing=Preformatted(numbered_lines(lines[start-1:end],start), STYLES['Mono'])
            story.append(KeepTogether([heading,listing]))
            for node in nodes:
                if start <= node.lineno <= end:
                    explanation = LINE_NOTES.get(name, {}).get(node.lineno) or explain(node)
                    if explanation:
                        last = statement_end(node)
                        label = f'L{node.lineno}' if last==node.lineno else f'L{node.lineno}–{last}'
                        story.append(para(label+': '+explanation, 'Note'))
    return manifest


def evidence_appendix(story, manifest):
    story.extend([PageBreak(),para('Evidence and source fingerprints','Chapter'),
                  para('The following source manifest binds the numbered listing to actual application files. Evidence JSON snippets below are copied from the committed compact archives. No training or evaluation is rerun by this generator.')])
    for item in manifest:
        story.append(para(item['file']+f" ({item['lines']} lines)", 'Sub'))
        story.append(Preformatted(item['sha256'],STYLES['Mono']))
    for relative in ['hmm_tuning_001/decoder_configuration.json', 'evaluator_test_rf_hmm_tuned_001/metrics.json','hmm_tuning_001/student_validation.json']:
        story.extend([PageBreak(),para(relative,'Chapter')])
        text=json.dumps(read_json(EVIDENCE/relative),indent=2)
        story.append(Preformatted(text,STYLES['Mono']))
    story.extend([PageBreak(),para('Verification map: test definitions','Chapter'),
                  para('These test names explain the checks implemented in the repository. The recorded earlier application run passed 51 cases, including parameterized variants. Listing a name is not a claim that this document-generation task reran model training or all tests.')])
    for path in sorted((ROOT/'tests').glob('test_*.py')):
        story.append(para(str(path.relative_to(ROOT)),'Sub'))
        tree=ast.parse(path.read_text())
        for node in tree.body:
            if isinstance(node,ast.FunctionDef) and node.name.startswith('test_'):
                story.append(para(f'L{node.lineno}: '+node.name,'Note'))
    record={'application_commit':subprocess.check_output(['git','rev-parse',SOURCE_SNAPSHOT],cwd=ROOT,text=True).strip(),
            'document':OUTPUT.name,'source_files':manifest,
            'all_application_source_lines':sum(item['lines'] for item in manifest),
            'manuscript_sha256':hashlib.sha256(MANUSCRIPT.read_bytes()).hexdigest(),
            'scope':'Complete src/pathsnap Python source, statement-level notes, project narrative and archived evidence'}
    (ROOT/'docs/full_project_guide_manifest.json').write_text(json.dumps(record,indent=2)+'\n')


def main():
    story=[Spacer(1,65),para('PathSnap','CoverTitle'),
           para('Complete project explanation\n& numbered code walkthrough','CoverTitle'),
           para('From noisy GPS points to a directed road route','CoverSub'),
           para('Random Forest · HMM/Viterbi · training and evaluation · shared backend · two-client integration'),
           Spacer(1,20), table(['Measured result','Value'],[
               ['Repeat evaluator pointwise accuracy','90.79%'],
               ['Student-test pointwise accuracy','92.19%'],
               ['Original evaluator HMM accuracy','75.41%'],
               ['Runtime classifier','One 200-tree Random Forest'],
           ],[310,WIDTH-396]), Spacer(1,25),
           para('Accuracy is against valid derived point references. The evaluator score is a repeat development benchmark, not whole-route accuracy or a newly untouched final test.'),
           para('18 November 2026 | model-training | application snapshot c74e243','Note'),
           PageBreak(),para('Contents','Chapter')]
    toc=TableOfContents()
    toc.levelStyles=[ParagraphStyle('TOC',fontName='Body',fontSize=8,leading=12,spaceBefore=4,leftIndent=0,firstLineIndent=0)]
    story.append(toc)
    markdown(story)
    manifest=append_source(story)
    evidence_appendix(story,manifest)
    doc=GuideDoc(OUTPUT)
    doc.multiBuild(story)
    print(f'Created {OUTPUT}\nPages: {doc.page}\nApplication source files: {len(manifest)}\nSource lines: {sum(m["lines"] for m in manifest)}')


if __name__=='__main__':
    main()
