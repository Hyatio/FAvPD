import sys
sys.path.append('.')
from process.utils.LogME import LogME

def logme_score(args, features, targets):
    if args.task_name == 'OpenEntity' or args.task_name == 'FIGER':
        logme = LogME(regression=True)
    elif args.task_name == 'TACRED' or args.task_name == 'FewRel' \
        or args.task_name == 'CHEMPROT' or args.task_name == 'FewNERD'\
            or args.task_name == 'ReTACRED':
        logme = LogME(regression=False)
    else:
        raise ValueError("wrong task name")
    score = logme.fit(features, targets)
    return score