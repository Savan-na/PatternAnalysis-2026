"""Three predeclared, isolated v3-based training recipes; no dataset access."""
import math
from pathlib import Path
import json

ARMS = ('C', 'L', 'S')
ROOT = Path(__file__).resolve().parents[2]
ORIGINAL_WEIGHTS = [1.453871, 1., 2.617199, 4.498586, 8.893215, 9.69995]
REFERENCE_SHA = '921cdb398ccde5c077c9e0a115ff8975b5c23bf7301aa2b19be40a0abcfda0e4'


def expected_config(arm):
    if arm not in ARMS:
        raise ValueError('Unknown branch')
    config = json.loads((ROOT/'code/config/v3_gn1_frozen.json').read_text(encoding='utf-8'))
    config.update(status='frozen_v5_v3_single_factor_screen',
                  purpose='Concurrent v3 control, small-organ CE compression, or Adam-only cosine',
                  models=[f'improved_3d_v5_{arm.lower()}'], branch=arm,
                  reference_manifest_sha256=REFERENCE_SHA)
    config['freeze_basis'] = 'V5_PARALLEL_SCREEN_ZH.md; v3 C/L/S single-factor screen'
    config['objective']['weight_source_scope'] = 'Original Train-only exposure weights; L applies a declared square-root compression to labels 3..5 only'
    config['objective']['ce_class_weights'] = class_weights(arm)
    config['optimizer'] = {'name': 'Adam', 'learning_rate': .001, 'weight_decay': 0.,
                           'schedule': 'cosine_6500_min_1e-5' if arm == 'S' else 'constant'}
    config['decision_criteria'] = {
        'selection': 'Original Validation patient-equal small-organ Dice; ten measured candidates; earliest tie',
        'primary_comparison': 'L minus C and S minus C; historical v3 is supplementary',
        'test': 'No Test decoding, selection, ranking or tuning during this screen',
        'interpretation': 'One seed, two screened interventions; winner requires replication',
        'secondary': 'All six classes, precision/recall, foreground surfaces and far Body FP; report regressions'}
    return config


def validate_config(config):
    if config != expected_config(config.get('branch')):
        raise ValueError('Config differs from the declared v3-based branch')


def class_weights(arm):
    if arm not in ARMS:
        raise ValueError('Unknown branch')
    weights = ORIGINAL_WEIGHTS.copy()
    if arm == 'L':
        weights[3:] = [round(math.sqrt(w), 6) for w in weights[3:]]
    return weights


def learning_rate(arm, updates):
    if arm not in ARMS or not isinstance(updates, int) or not 0 <= updates <= 6500:
        raise ValueError('Invalid branch/update count')
    return 1e-5 + .5*(.001-1e-5)*(1+math.cos(math.pi*updates/6500)) if arm == 'S' else .001


def scheduler_state(arm, updates):
    learning_rate(arm, updates)
    return {'branch': arm, 'completed_updates': updates,
            'kind': 'cosine_6500_min_1e-5' if arm == 'S' else 'constant'}


def make_optimizer(model, arm):
    import torch
    return torch.optim.Adam(model.parameters(), lr=learning_rate(arm, 0), weight_decay=0.)


def loss_components(logits, target, weights):
    from torch.nn import functional as F
    from .losses import foreground_soft_dice_loss
    ce = F.cross_entropy(logits, target, weight=weights)
    dice = foreground_soft_dice_loss(logits, target)
    return .5*ce+.5*dice, ce.detach(), dice.detach()


def restore_state(model, optimizer, checkpoint, config_sha, model_id, arm):
    """Check branch, schedule and cursor before restoring model/optimizer/RNG."""
    import numpy as np
    import torch
    if checkpoint['config_sha256'] != config_sha or checkpoint['model'] != model_id:
        raise ValueError('Recovery config/model mismatch')
    e, b = checkpoint['epoch'], checkpoint['next_batch']
    if not isinstance(e, int) or not isinstance(b, int) or not 0 <= e <= 50 or not 0 <= b <= 130:
        raise ValueError('Recovery cursor invalid')
    updates = e*130+b
    if updates > 6500 or checkpoint['completed_updates'] != updates:
        raise ValueError('Recovery update count invalid')
    if checkpoint['scheduler_state'] != scheduler_state(arm, updates):
        raise ValueError('Recovery scheduler mismatch')
    saved_optimizer = checkpoint['optimizer_state']
    if any(g['lr'] != learning_rate(arm, updates) or g.get('weight_decay') != 0.
           for g in saved_optimizer['param_groups']):
        raise ValueError('Recovery learning rate or decay mismatch')
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(saved_optimizer)
    torch.set_rng_state(checkpoint['torch_rng_state'].cpu())
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in checkpoint['cuda_rng_states']])
    np.random.set_state(checkpoint['numpy_rng_state'])
