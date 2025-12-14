from copy import deepcopy
from typing import Any, Optional, Union


MAX_CONSTANTS: int = 10
PROP: str = "pqrs"
VAR: str = "abcdefghijxyzw"
PRED: str = "PQRS"
NEG: str = '~'
UNARY_CONN: list[str] = ['E', 'A']
BINARY_CONN: list[str] = ['&', '\\/', '->']
DEBUG: bool = False


class Var:   
    def __init__(self, var: Optional[str]) -> None:
        if var is None or len(var) != 1 or var not in VAR:
            raise ValueError()
        self.val: str = var
    
    def __repr__(self) -> str:
        return self.val
    
    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Var) and self.val == other.val


class Prop:    
    def __init__(self, prop: Optional[str]) -> None:
        if prop is None or len(prop) != 1 or prop not in PROP:
            raise ValueError()
        self.val: str = prop
        self.is_fol: bool = False
    
    def __repr__(self) -> str:
        return self.val
    
    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Prop) and self.val == other.val
    
    def isWrong(self) -> bool:
        return False
    
    def replaceVar(self, orig_var: Var, new_val: str) -> None:
        return


class Negation:  
    def __init__(self, a: Union[str, Formula]) -> None:
        self.a: Formula = Formula(a) if type(a) is str else a
        self.is_fol: bool = self.a.is_fol
        self._str: Optional[str] = None
    
    def __repr__(self) -> str:
        if self._str is None:
            self._str = f"~{self.a}"
        return self._str
    
    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Negation) and self.a == other.a
    
    def isWrong(self) -> bool:
        return self.a.isWrong()
    
    def replaceVar(self, orig_var: Var, new_val: str) -> None:
        self.a.replaceVar(orig_var, new_val)
        self._str = None


class BinaryConn:    
    def __init__(self, conn: Optional[str]) -> None:
        if conn is None or conn not in BINARY_CONN:
            raise ValueError()
        self.val: str = conn
    
    def __repr__(self) -> str:
        return self.val


class BinaryOp:    
    def __init__(self, a: str, conn: str, b: str) -> None:
        self.a: Formula = Formula(a)
        self.conn: BinaryConn = BinaryConn(conn)
        self.b: Formula = Formula(b)
        self.is_fol: bool = self.a.is_fol or self.b.is_fol
        self._str: Optional[str] = None
    
    def __repr__(self) -> str:
        if self._str is None:
            self._str = f"({self.a}{self.conn}{self.b})"
        return self._str
    
    def __eq__(self, other: Any) -> bool:
        return isinstance(other, BinaryOp) and self.conn == other.conn and self.a == other.a and self.b == other.b
    
    def isWrong(self) -> bool:
        return self.a.isWrong() or self.b.isWrong()
    
    def replaceVar(self, orig_var: Var, new_val: str) -> None:
        self.a.replaceVar(orig_var, new_val)
        self.b.replaceVar(orig_var, new_val)
        self._str = None


class UnaryConn:    
    def __init__(self, conn: Optional[str]) -> None:
        if conn is None or conn not in UNARY_CONN:
            raise ValueError()
        self.val: str = conn
    
    def __repr__(self) -> str:
        return self.val


class UnaryOp:    
    def __init__(self, conn: str, var: str, a: str) -> None:
        self.conn: UnaryConn = UnaryConn(conn)
        self.var: Var = Var(var)
        self.a: Formula = Formula(a)
        self.is_fol: bool = True
        self._str: Optional[str] = None
    
    def __repr__(self) -> str:
        if self._str is None:
            self._str = f"{self.conn}{self.var}{self.a}"
        return self._str
    
    def __eq__(self, other: Any) -> bool:
        return isinstance(other, UnaryOp) and self.conn == other.conn and self.var == other.var and self.a == other.a
    
    def isWrong(self) -> bool:
        return self.a.isWrong()
    
    def replaceVar(self, orig_var: Var, new_val: str) -> None:
        if orig_var != self.var:
            self.a.replaceVar(self.var, new_val)
            self._str = None


class Pred:    
    def __init__(self, pred: Optional[str]) -> None:
        if pred is None or len(pred) != 1 or pred not in PRED:
            raise ValueError()
        self.val: str = pred
    
    def __repr__(self) -> str:
        return self.val
    
    def __eq__(self, other) -> bool:
        return isinstance(other, Pred) and self.val == other.val
    
    def replaceVar(self, orig_var: Var, new_val: str) -> None:
        return


class Atom:    
    def __init__(self, pred: str, a: str, b: str) -> None:
        self.pred: Pred = Pred(pred)
        self.a: Var = Var(a)
        self.b: Var = Var(b)
        self.is_fol: bool = True
        self._str: Optional[str] = None
    
    def __repr__(self) -> str:
        if self._str is None:
            self._str = f"{self.pred}({self.a},{self.b})"
        return self._str
    
    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Atom) and self.pred == other.pred and self.a == other.a and self.b == other.b
    
    def isWrong(self) -> bool:
        return False
    
    def replaceVar(self, orig_var: Var, new_val: str) -> None:
        if self.a == orig_var:
            self.a.val = new_val
            self._str = None
        if self.b == orig_var:
            self.b.val = new_val
            self._str = None


TFormula = Union[Prop, Negation, UnaryOp, BinaryOp, Atom]
Theory = tuple[list[TFormula], str, dict[str, str]]     # (Formulas, constants, gamma memory)
Exp = tuple[str, TFormula, Union[TFormula, str, None]]


class Formula:    
    def __init__(self, fmla: str) -> None:
        self.fmla: Optional[TFormula] = None
        self.fmla_type: Optional[int] = None
        self.is_fol: bool = self.fmla.is_fol if self.fmla is not None else False
        self.is_prop: bool = self.fmla.is_prop if self.fmla is not None else False
        self.lhs: str = ""
        self.conn: str = ""
        self.rhs: str = ""
        self._str: Optional[str] = None
        
        try:
            if not fmla:
                return
                
            if fmla[0] != '(':
                if len(fmla) == 1:
                    self.fmla = Prop(fmla)
                if fmla[0] == NEG:
                    self.fmla = Negation(fmla[1:])
                if fmla[0] == 'A':
                    self.fmla = UnaryOp('A', fmla[1], fmla[2:])
                if fmla[0] == 'E':
                    self.fmla = UnaryOp('E', fmla[1], fmla[2:])
                if fmla[0] in PRED and fmla[1] == '(' and fmla[2] in VAR and fmla[3] == ',' and fmla[4] in VAR and fmla[5] == ')':
                    self.fmla = Atom(fmla[0], fmla[2], fmla[4])
            else:
                op_br: int = 0
                for i in range(len(fmla)):
                    if fmla[i] == '(':
                        op_br += 1
                    elif fmla[i] == ')':
                        op_br -= 1
                    elif op_br == 1:
                        for c in BINARY_CONN:
                            if fmla[i:i+len(c)] == c:
                                self.lhs = fmla[1:i]
                                self.conn = c
                                self.rhs = fmla[i+len(c):-1]
                                self.fmla = BinaryOp(self.lhs, self.conn, self.rhs)
                                
            self.is_fol = self.fmla.is_fol if self.fmla is not None else False
            self.is_prop = self.fmla.is_prop if self.fmla is not None else False
        except Exception as e:
            print(f"Failed:\n{e}")
            self.fmla = None
    
    def __repr__(self) -> str:
        if self._str is None:
            self._str = str(self.fmla)
        return self._str
    
    def isWrong(self) -> bool:
        if self.fmla is None:
            return True
        return self.fmla.isWrong()
    
    def replaceVar(self, orig_var: Var, new_val: str) -> None:
        self.fmla.replaceVar(orig_var, new_val)
        self._str = None
    
    def getType(self) -> int:
        if self.fmla_type is not None:
            return self.fmla_type
            
        fmla_type = type(self.fmla)
        
        if (self.is_fol and self.is_prop) or self.fmla is None or self.isWrong():
            self.fmla_type = 0
        elif fmla_type is Atom:
            self.fmla_type = 1
        elif fmla_type is Negation:
            self.fmla_type = 2 if self.is_fol else 7
        elif fmla_type is UnaryOp:
            if self.fmla.conn.val == 'A':
                self.fmla_type = 3
            if self.fmla.conn.val == 'E':
                self.fmla_type = 4
        elif fmla_type is BinaryOp:
            self.fmla_type = 5 if self.is_fol else 8
        elif fmla_type is Prop:
            self.fmla_type = 6
        
        if self.fmla_type is None:
            raise Exception(f"This shouldn't happen: {self.fmla}")
        return self.fmla_type
    
    def isSat(self) -> int:
        if self.fmla is None:
            return 0
        
        tab = [([self.fmla], "", {})]
        if DEBUG: print("")
        while tab:
            if DEBUG: self._printTab(tab)
            
            sigma, constants, gamma_used = tab.pop(0)
            if self._isExp(sigma) and not self._closes(sigma):
                return 1
            if len(constants) >= MAX_CONSTANTS:
                return 2
            
            # Pick first non-literal in sigma (or return sat if there are none)
            exp_type = exp_a = exp_b = psi_i = None
            for i in range(len(sigma)):
                psi = sigma[i]
                curr_exp_type, curr_exp_a, curr_exp_b = self._expNode(psi)
                if curr_exp_type == "alpha":
                    exp_type, exp_a, exp_b = curr_exp_type, curr_exp_a, curr_exp_b
                    psi_i = i
                    break
                elif curr_exp_type == "beta" and (exp_type is None or exp_type == "gamma"):
                    exp_type, exp_a, exp_b = curr_exp_type, curr_exp_a, curr_exp_b
                    psi_i = i
                elif curr_exp_type == "delta":
                    exp_type, exp_a, exp_b = curr_exp_type, curr_exp_a, curr_exp_b
                    psi_i = i
                elif curr_exp_type == "gamma":
                    if exp_type is None:
                        exp_type, exp_a, exp_b = curr_exp_type, curr_exp_a, curr_exp_b
                        psi_i = i
                    elif exp_type == "gamma":
                        if gamma_used.get(str(psi), "") < gamma_used.get(str(sigma[psi_i]), ""):
                            exp_type, exp_a, exp_b = curr_exp_type, curr_exp_a, curr_exp_b
                            psi_i = i
            
            if exp_type is None:
                return 1
            
            if DEBUG: print(f"Expansion: ({exp_type}, {exp_a}, {exp_b})")
            
            if exp_type == "alpha":
                sigma.pop(psi_i)
                sigma.append(exp_a)
                if exp_b is not None:
                    sigma.append(exp_b)
                if not self._closes(sigma) and sigma not in tab:
                    tab.append((sigma, constants, gamma_used))
            elif exp_type == "beta":
                sigma.pop(psi_i)
                sigma1 = sigma + [exp_a]
                sigma2 = sigma + [exp_b]
                if not self._closes(sigma1) and sigma1 not in tab:
                    tab.append((sigma1, constants, gamma_used.copy()))
                if not self._closes(sigma2) and sigma2 not in tab:
                    tab.append((sigma2, constants, gamma_used.copy()))
            elif exp_type == "delta":
                sigma.pop(psi_i)
                exp_a.replaceVar(exp_b, VAR[len(constants)])
                sigma.append(exp_a)
                constants += VAR[len(constants)]
                if not self._closes(sigma) and sigma not in tab:
                    tab.append((sigma, constants, gamma_used))
            elif exp_type == "gamma":
                if not constants:
                    constants = "a"
                    gamma_used[str(sigma[psi_i])] = "a"
                    exp_a = deepcopy(exp_a)
                    exp_a.replaceVar(exp_b, "a")
                    sigma.append(exp_a)
                    tab.append((sigma, constants, gamma_used))
                    continue
                
                for c in constants:
                    exp_a_str = str(sigma[psi_i])
                    if exp_a_str not in gamma_used or c not in gamma_used[exp_a_str]:
                        gamma_used[exp_a_str] = gamma_used.get(exp_a_str, "") + c
                        exp_a = deepcopy(exp_a)
                        exp_a.replaceVar(exp_b, c)
                        sigma.append(exp_a)
                        tab.append((sigma, constants, gamma_used))
                        break
                else:
                    return 1
            
            if DEBUG: print("")
        
        return 0
    
    def _printTab(self, tab: list[Theory]):
        for i in range(len(tab)):
            print(f"Branch {i}")
            for fmla in tab[i][0]:
                print(f"\t- {fmla}")
            if tab[i][1]:
                print(f"\tConstants: {tab[i][1] or 'None'}")
            if tab[i][2]:
                print("\tUsed consts gamma:")
                for gamma, consts in tab[i][2].items():
                    print(f"\t\t- {gamma}: {consts}")
    
    def _isExp(self, sigma: list[TFormula]) -> bool:
        for f in sigma:
            if isinstance(f, Formula):
                f = f.fmla
            if isinstance(f, Negation):
                f = f.a.fmla
            if type(f) not in (Prop, Var):
                return False
        return True
    
    def _closes(self, sigma: list[TFormula]) -> bool:
        positive: set[str] = set()
        negative: set[str] = set()
        
        for f in sigma:
            if isinstance(f, Negation):
                inner = str(f.a)
                if inner in positive:
                    return True
                negative.add(inner)
            else:
                f_str = str(f)
                if f_str in negative:
                    return True
                positive.add(f_str)
                
        return False
    
    def _expNode(self, psi: TFormula):
        psi_type = type(psi)
        
        if psi_type is BinaryOp:
            if psi.conn.val == '&':
                return ("alpha", psi.a.fmla, psi.b.fmla)
            if psi.conn.val == '->':
                return ("beta", Negation(psi.a), psi.b.fmla)
            return ("beta", psi.a.fmla, psi.b.fmla)
        if psi_type is UnaryOp:
            if psi.conn.val == 'E':
                return ("delta", psi.a.fmla, psi.var)
            return ("gamma", psi.a.fmla, psi.var)
        if psi_type is Negation:
            f = psi.a.fmla
            f_type = type(f)
            if f_type is BinaryOp:
                if f.conn.val == '&':
                    return ("beta", Negation(f.a), Negation(f.b))
                if f.conn.val == '->':
                    return ("alpha", f.a.fmla, Negation(f.b))
                return ("alpha", Negation(f.a), Negation(f.b))
            if f_type is Negation:
                return ("alpha", f.a.fmla, None)
            if f_type is UnaryOp:
                if f.conn.val == 'E':
                    return ("gamma", Negation(f.a), f.var)
                return ("delta", Negation(f.a), f.var)
                
        return (None, None, None)


if __name__ == "__main__":
    formula: Formula = Formula("")
    
    with open("input.txt") as f:
        parseOutputs = ["not a formula",
                        "an atom",
                        "a negation of a first order logic formula",
                        "a universally quantified formula",
                        "an existentially quantified formula",
                        "a binary connective first order formula",
                        "a proposition",
                        "a negation of a propositional formula",
                        "a binary connective propositional formula"]
        satOutput = ["is not satisfiable", "is satisfiable", "may or may not be satisfiable"]
        
        firstline = f.readline()
        PARSE = "PARSE" in firstline
        SAT = "SAT" in firstline
        
        for line in f:
            if line[-1] == '\n':
                line = line[:-1]
            parsed = parse(line)
        
            if PARSE:
                output = f"{line} is {parseOutputs[parsed]}."
                if parsed in (5,8):
                    output += f" Its left hand side is {lhs(line)}, its connective is {con(line)}, and its right hand side is {rhs(line)}."
                print(output)
        
            if SAT:
                if parsed:
                    tableau = [theory(line)]
                    print(f"{line} {satOutput[sat(tableau)]}")
                else:
                    print(f"{line} is not a formula")
