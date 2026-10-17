"""Six-in-a-row window evaluation with turn-aware tactical and bounded pair search."""
from itertools import combinations
from collections import OrderedDict
import time
import random
import numpy as np
from game.connect6_env import Player, Connect6Env

# Every length-six segment, including segments with gaps between stones.
_WINDOWS = np.array([
    [(r+i*dr)*19+c+i*dc for i in range(6)]
    for r in range(19) for c in range(19)
    for dr,dc in ((0,1),(1,0),(1,1),(1,-1))
    if 0 <= r+5*dr < 19 and 0 <= c+5*dc < 19
], dtype=np.int16)
_MEMBERSHIP = np.zeros((len(_WINDOWS),361),dtype=np.int8)
_MEMBERSHIP[np.arange(len(_WINDOWS))[:,None],_WINDOWS] = 1
_WEIGHTS = np.array([0,1,8,70,700,9000,10000000],dtype=float)
_CENTER = 18-np.abs(np.arange(361)//19-9)-np.abs(np.arange(361)%19-9)


class HeuristicAI:
    """Tactical wins/defence are exhaustive; quiet search uses a limited candidate beam.

    noise_prob is for data generation and applies only when no urgent threat exists.
    """
    BOARD_SIZE = 19

    def __init__(self, noise_prob=0.1, beam_width=12, reply_width=6,
                 threat_depth=3, threat_nodes=384, threat_seconds=1.0):
        self.noise_prob = noise_prob
        self.beam_width = max(2,int(beam_width))
        self.reply_width = max(2,int(reply_width))
        self.threat_depth = max(1,int(threat_depth))
        self.threat_nodes = max(0,int(threat_nodes))
        self.threat_seconds = max(0,float(threat_seconds))
        self._action_cache = OrderedDict()
        self._planned_moves = OrderedDict()
        self._proof_cache = OrderedDict()
        self._threat_cache = OrderedDict()
        self._known_wins = OrderedDict()
        self._nodes_left = self.threat_nodes
        self._deadline = float('inf')
        self._decision_deadline = float('inf')
        self.last_proof = None

    @staticmethod
    def _threats(flat, player, stones=2):
        values=flat[_WINDOWS]
        counts=np.sum(values==player,axis=1)
        possible=(np.sum(values==-player,axis=1)==0)&(counts>=6-stones)&(counts<6)
        return sorted({frozenset(map(int,window[row==0]))
                       for window,row in zip(_WINDOWS[possible],values[possible])},
                      key=lambda points:(len(points),tuple(sorted(points))))

    @staticmethod
    def _covers(threats, budget):
        """Find all minimal blocking sets of up to budget stones (budget <= 2)."""
        if not threats:
            return [frozenset()]
        if budget==0:
            return []
        answers=set()
        for point in sorted(min(threats,key=len)):
            remaining=[t for t in threats if point not in t]
            for rest in HeuristicAI._covers(remaining,budget-1):
                answers.add(rest|{point})
        return sorted(answers,key=lambda points:(len(points),tuple(sorted(points))))

    @staticmethod
    def _counts(flat,player):
        values=flat[_WINDOWS]
        return np.sum(values==player,axis=1),np.sum(values==-player,axis=1)

    @staticmethod
    def _value(own,other):
        return float(np.sum(_WEIGHTS[own]*(other==0))-1.15*np.sum(_WEIGHTS[other]*(own==0)))

    @staticmethod
    def _point_scores(own,other):
        attack=(_WEIGHTS[np.minimum(own+1,6)]-_WEIGHTS[own])*(other==0)
        defence=(_WEIGHTS[np.minimum(other+1,6)]-_WEIGHTS[other])*(own==0)
        return (attack+1.15*defence)@_MEMBERSHIP + _CENTER*.05

    def _ranked(self,flat,scores,limit):
        legal=np.flatnonzero(flat==0)
        # Stable order makes a zero-noise opponent reproducible.
        return sorted(map(int,legal),key=lambda a:(-scores[a],a))[:limit]

    @staticmethod
    def _remember(cache,key,value):
        cache[key]=value
        cache.move_to_end(key)
        if len(cache)>2048:
            cache.popitem(last=False)

    def get_action(self, env: Connect6Env) -> int:
        flat=env.board.ravel()
        key=(flat.tobytes(),int(env.current_player),env.get_remaining_moves())
        self.last_proof=None
        if key in self._planned_moves:
            action=self._planned_moves[key]
            if 0<=action<361 and flat[action]==0:
                return int(action)
        if self.noise_prob==0 and key in self._action_cache:
            action=self._action_cache[key]
            if 0<=action<361 and flat[action]==0:
                self._action_cache.move_to_end(key)
                return int(action)
        self._decision_deadline=time.monotonic()+self.threat_seconds
        try:
            action=int(self._choose_action(env))
        finally:
            self._decision_deadline=float('inf')
        if self.noise_prob==0 and action>=0:
            self._remember(self._action_cache,key,action)
        return action

    def _start_plan(self,flat,player,budget,plan,scores):
        threats=self._cached_threats(flat,-player,2)
        first=max(plan,key=lambda a:(sum(a in t for t in threats),scores[a],-a))
        if len(plan)==2 and budget==2:
            after=flat.copy();after[first]=player
            second=next(a for a in plan if a!=first)
            self._remember(self._planned_moves,(after.tobytes(),player,1),second)
        return int(first)

    def _cached_threats(self,flat,player,stones=2):
        key=(flat.tobytes(),int(player),stones)
        if key not in self._threat_cache:
            self._remember(self._threat_cache,key,tuple(self._threats(flat,player,stones)))
        else:
            self._threat_cache.move_to_end(key)
        return self._threat_cache[key]

    def _defence_plans(self,flat,threats,limit=64):
        legal=list(map(int,np.flatnonzero(flat==0)))
        budget=min(2,len(legal))
        covers=self._covers(threats,budget)
        answers=set()
        for cover in covers:
            free=[a for a in legal if a not in cover]
            for extra in combinations(free,budget-len(cover)):
                answers.add(tuple(sorted((*cover,*extra))))
                if len(answers)>limit:
                    return None
        return sorted(answers)

    def _pair_plans(self,flat,player,own,other,ranked):
        # A useful partner can be weak until the first stone changes the position.
        plans=set(combinations(ranked,2))
        for first in ranked:
            trial=flat.copy();trial[first]=player
            scores=self._point_scores(own+_MEMBERSHIP[:,first],other)
            for second in self._ranked(trial,scores,3):
                plans.add(tuple(sorted((first,second))))
        return sorted(plans)

    def _forcing_plans(self,flat,player,budget,limit):
        own,other=self._counts(flat,player)
        scores=self._point_scores(own,other)
        ranked=self._ranked(flat,scores,self.beam_width)
        threats=self._cached_threats(flat,-player,2)
        covers=self._covers(threats,budget)
        if not covers:
            return []
        plans=set()
        if budget==1:
            plans={(a,) for a in ranked} if not threats else {tuple(sorted(c)) for c in covers}
        elif threats:
            for cover in covers:
                if len(cover)==2:
                    plans.add(tuple(sorted(cover)))
                else:
                    plans.update(tuple(sorted((*cover,a))) for a in ranked if a not in cover)
        else:
            plans=set(self._pair_plans(flat,player,own,other,ranked))
        return sorted(plans,key=lambda plan:(-self._value(own+_MEMBERSHIP[:,list(plan)].sum(axis=1),other),plan))[:limit]

    def _reply_plans(self,flat,ranked,threats):
        # Forced blocking points must not disappear outside the static beam.
        if not threats:
            return list(combinations(ranked,2)) if len(ranked)>1 else [(a,) for a in ranked]
        plans=set()
        budget=min(2,int(np.count_nonzero(flat==0)))
        for cover in self._covers(threats,budget):
            if len(cover)==budget:
                plans.add(tuple(sorted(cover)))
            else:
                plans.update(tuple(sorted((*cover,a))) for a in ranked if a not in cover)
        return sorted(plans)

    def _find_forced_win(self,flat,player,budget,depth=None):
        self._nodes_left=self.threat_nodes
        self._deadline=min(time.monotonic()+self.threat_seconds,self._decision_deadline)
        return self._prove_win(flat,int(player),budget,depth or self.threat_depth)

    def _limited(self):
        return self._nodes_left<=0 or time.monotonic()>=self._deadline

    def _prove_win(self,flat,player,budget,depth):
        key=(flat.tobytes(),player,budget,depth)
        if key in self._proof_cache:
            self._proof_cache.move_to_end(key)
            return self._proof_cache[key]
        known=(flat.tobytes(),player,budget)
        if known in self._known_wins:
            return self._known_wins[known]
        if self._limited():
            return None
        self._nodes_left-=1
        wins=self._cached_threats(flat,player,budget)
        if wins:
            plan=tuple(sorted(wins[0]))
            self._remember(self._proof_cache,key,plan)
            self._remember(self._known_wins,known,plan)
            return plan
        unknown=False
        for plan in self._forcing_plans(flat,player,budget,24 if depth==self.threat_depth else 8):
            if self._limited():
                return None
            result=self._prove_plan(flat,player,plan,depth)
            if result is True:
                self._remember(self._proof_cache,key,plan)
                self._remember(self._known_wins,known,plan)
                return plan
            unknown=unknown or result is None
        if unknown:
            return None
        self._remember(self._proof_cache,key,False)
        return False

    def _prove_plan(self,flat,player,plan,depth):
        trial=flat.copy();trial[list(plan)]=player
        own,other=self._counts(trial,player)
        if np.any((own==6)&(other==0)):
            return True
        # Defender moves first; even a fork loses to an immediate counter-win.
        if self._cached_threats(trial,-player,2):
            return False
        threats=self._cached_threats(trial,player,2)
        if not threats:
            return False
        replies=self._defence_plans(trial,threats)
        if replies is None:
            return None
        if not replies:
            return True
        if depth<=1:
            return False
        unknown=False
        for reply in replies:
            if self._limited():
                return None
            defended=trial.copy();defended[list(reply)]=-player
            budget=min(2,int(np.count_nonzero(defended==0)))
            if budget==0:
                return False
            continuation=self._prove_win(defended,player,budget,depth-1)
            if continuation is False:
                return False
            unknown=unknown or continuation is None
        return None if unknown else True

    def _avoid_forced_loss(self,flat,player,plans):
        # Only a proved opponent win rejects a move; unknown is not a loss.
        if self.threat_depth<=1 or np.count_nonzero(flat==-player)<3:
            return plans[0]
        for plan in plans[:4]:
            trial=flat.copy();trial[list(plan)]=player
            reply_budget=min(2,int(np.count_nonzero(trial==0)))
            if reply_budget==0:
                return plan
            attack=self._find_forced_win(trial,-player,reply_budget)
            if not isinstance(attack,tuple):
                return plan
        return plans[0]

    def _choose_action(self, env: Connect6Env) -> int:
        flat=env.board.ravel().copy()
        legal=np.flatnonzero(flat==0)
        if not len(legal):
            return -1
        if not np.any(flat):
            return 180
        player=int(env.current_player)
        budget=min(env.get_remaining_moves(),len(legal))
        own,other=self._counts(flat,player)
        scores=self._point_scores(own,other)
        wins=self._threats(flat,player,budget)
        if wins:
            # The chosen segment can be completed before the opponent gets a turn.
            return min(wins[0],key=lambda a:(-scores[a],a))

        threats=self._threats(flat,-player,2)
        covers=self._covers(threats,budget)
        if threats and not covers:
            # A forced loss: block as many distinct winning combinations as possible.
            return max(map(int,legal),key=lambda a:(sum(a in t for t in threats),scores[a],-a))
        if not threats and random.random()<self.noise_prob:
            candidates=self._get_candidates(env.board)
            r,c=random.choice(candidates)
            return r*19+c

        ranked=self._ranked(flat,scores,self.beam_width)
        if budget==1:
            plans=[(a,) for a in ranked] if not threats else [tuple(c) for c in covers]
        elif threats:
            plans=set()
            for cover in covers:
                if len(cover)==2:
                    plans.add(tuple(sorted(cover)))
                else:
                    for a in ranked:
                        if a not in cover:
                            plans.add(tuple(sorted((*cover,a))))
            plans=sorted(plans)
        else:
            plans=self._pair_plans(flat,player,own,other,ranked)

        best_plan=(ranked[0],)
        best_score=-float('inf')
        scored_plans=[]
        for plan in plans:
            delta=_MEMBERSHIP[:,list(plan)].sum(axis=1)
            after_own=own+delta
            trial=flat.copy();trial[list(plan)]=player
            own_threats=self._threats(trial,player,2)
            if self._threats(trial,-player,2):
                value=-1e9
            elif own_threats and not self._covers(own_threats,2):
                # Opponent cannot block all winning segments with its two stones.
                value=1e9
            else:
                # Let the opponent choose its best pair from a smaller beam.
                reply_scores=self._point_scores(other,after_own)
                replies=self._ranked(trial,reply_scores,self.reply_width)
                reply_plans=self._reply_plans(trial,replies,own_threats)
                value=float('inf')
                for reply in reply_plans:
                    reply_delta=_MEMBERSHIP[:,list(reply)].sum(axis=1)
                    after_other=other+reply_delta
                    # Reward attack structure that survives the opponent's reply.
                    residual=float(np.sum(_WEIGHTS[after_own]*((after_other==0)&(after_own>=2))))
                    value=min(value,self._value(after_own,after_other)+.15*residual)
                if not replies:
                    value=self._value(after_own,other)
            value+=sum(_CENTER[a] for a in plan)*.01
            scored_plans.append((value,plan))
            if value>best_score:
                best_score=value
                best_plan=plan
        if self.threat_depth > 1 and np.count_nonzero(flat==player)>=3:
            plan=self._find_forced_win(flat,player,budget,depth=self.threat_depth)
            if isinstance(plan,tuple):
                self.last_proof={'plan':plan,'depth':self.threat_depth}
                return self._start_plan(flat,player,budget,plan,scores)
        if best_score<1e8:
            ordered=[plan for value,plan in sorted(scored_plans,key=lambda item:-item[0])]
            best_plan=self._avoid_forced_loss(flat,player,ordered)
        return self._start_plan(flat,player,budget,best_plan,scores)

    def _get_candidates(self,board):
        occupied=np.argwhere(board!=Player.EMPTY)
        candidates=set()
        for r,c in occupied:
            for dr in range(-2,3):
                for dc in range(-2,3):
                    nr,nc=int(r+dr),int(c+dc)
                    if 0<=nr<19 and 0<=nc<19 and board[nr,nc]==0:
                        candidates.add((nr,nc))
        return sorted(candidates) or [tuple(map(int,p)) for p in np.argwhere(board==0)]

    def _evaluate_point(self,board,x,y,player):
        if board[x,y]!=0:
            return 0.0
        flat=board.ravel().copy();action=x*19+y
        own,other=self._counts(flat,int(player))
        relevant=_MEMBERSHIP[:,action].astype(bool)
        # Sum every viable direction; blocked segments contribute zero.
        return float(np.sum((_WEIGHTS[own[relevant]+1]-_WEIGHTS[own[relevant]])*(other[relevant]==0)))
