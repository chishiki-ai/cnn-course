Search.setIndex({"docnames": ["README", "intro-building-cnn-pytorch-exercise", "intro-building-cnn-pytorch-solution", "lecture_note", "part2-building-cnn-pytorch-exercise", "part2-building-cnn-pytorch-solution"], "filenames": ["README.md", "intro-building-cnn-pytorch-exercise.ipynb", "intro-building-cnn-pytorch-solution.ipynb", "lecture_note.md", "part2-building-cnn-pytorch-exercise.ipynb", "part2-building-cnn-pytorch-solution.ipynb"], "titles": ["Building Scalable CNN models", "Building a CNN Classifier with PyTorch: Part 1", "Building a CNN Classifier with PyTorch: Part 1", "Image Classification and Convolutional Neural Network", "Building a CNN Classifier with PyTorch: Part 2", "Building a CNN Classifier with PyTorch: Part 2"], "terms": {"thi": [0, 1, 2, 3, 4, 5], "cours": [0, 3], "i": [0, 1, 2, 4, 5], "design": 0, "teach": 0, "you": [0, 1, 2, 4, 5], "how": [0, 1, 2, 3, 4, 5], "classifi": [0, 3], "interpret": 0, "The": [0, 1, 2, 3, 4, 5], "content": 0, "follow": [0, 1, 2, 3, 4, 5], "imag": [0, 1, 2], "classif": [0, 1, 2, 4, 5], "convolut": 0, "neural": 0, "network": 0, "pytorch": [0, 3], "part": 0, "1": [0, 3, 4, 5], "2": [0, 1, 2, 3], "In": [1, 2, 3, 4, 5], "tutori": [1, 2, 3, 4, 5], "we": [1, 2, 3], "walk": [1, 2, 3], "through": [1, 2, 3], "basic": [1, 2, 3], "compon": [1, 2], "us": [1, 2, 3, 4, 5], "from": [1, 2, 4, 5], "hurrican": [1, 2, 4, 5], "harvei": [1, 2, 4, 5], "categori": [1, 2, 4, 5], "4": [1, 2, 4, 5], "hit": [1, 2, 4, 5], "texa": [1, 2, 4, 5], "august": [1, 2, 4, 5], "2017": [1, 2, 3, 4, 5], "result": [1, 2, 3, 4, 5], "catastroph": [1, 2, 4, 5], "flood": [1, 2, 4, 5], "houston": [1, 2, 4, 5], "metropolitan": [1, 2, 4, 5], "area": [1, 2, 3, 4, 5], "data": [1, 2, 4, 5], "set": [1, 2, 3], "specif": [1, 2, 3, 4, 5], "focus": [1, 2, 4, 5], "home": [1, 2, 4, 5], "accord": [1, 2, 4, 5], "amount": [1, 2, 4, 5], "damag": [1, 2, 4, 5], "receiv": [1, 2, 4, 5], "all": [1, 2, 3, 4, 5], "ar": [1, 2, 3, 4, 5], "label": [1, 2, 3], "c0": [1, 2, 4, 5], "c2": [1, 2, 4, 5], "c4": [1, 2, 4, 5], "respect": [1, 2, 4, 5], "low": [1, 2, 4, 5], "medium": [1, 2, 3, 4, 5], "high": [1, 2, 4, 5], "popular": [1, 2, 3], "machin": [1, 2, 3, 4, 5], "librari": [1, 2], "deep": [1, 2, 4, 5], "develop": [1, 2, 3], "facebook": [1, 2], "can": [1, 2, 3, 4, 5], "broken": [1, 2], "down": [1, 2, 3], "let": [1, 2, 4, 5], "": [1, 2, 3, 4, 5], "get": [1, 2, 4, 5], "start": [1, 2, 4, 5], "import": [1, 2, 3, 4, 5], "modul": [1, 2, 4, 5], "need": [1, 2, 3, 4, 5], "notebook": [1, 2, 4, 5], "well": [1, 2, 3, 4, 5], "few": [1, 2, 4, 5], "hyperparamet": [1, 2, 4, 5], "throughout": [1, 2, 4, 5], "Then": [1, 2, 4, 5], "dive": [1, 2, 3], "torch": [1, 2, 4, 5], "torchvis": [1, 2, 4, 5], "nn": [1, 2, 4, 5], "datetim": [1, 2, 4, 5], "matplotlib": [1, 2, 4, 5], "pyplot": [1, 2, 4, 5], "plt": [1, 2, 4, 5], "inlin": [1, 2, 4, 5], "hub": [1, 2, 4, 5], "set_dir": [1, 2, 4, 5], "tmp": [1, 2, 4, 5], "remov": [1, 2, 4, 5], "when": [1, 2, 3, 4, 5], "run": [1, 2, 4, 5], "here": [1, 2, 3, 4, 5], "rate": [1, 2], "lr": [1, 2, 4, 5], "much": [1, 2, 4, 5], "paramet": [1, 2, 3, 4, 5], "updat": [1, 2, 3, 4, 5], "each": [1, 2, 3, 4, 5], "batch": [1, 2, 4, 5], "epoch": [1, 2, 4, 5], "size": [1, 2, 4, 5], "number": [1, 2, 3, 4, 5], "point": [1, 2, 4, 5], "estim": [1, 2, 4, 5], "gradient": [1, 2, 3, 4, 5], "iter": [1, 2, 3, 4, 5], "time": [1, 2, 3, 4, 5], "over": [1, 2, 3, 4, 5], "our": [1, 2, 3, 4, 5], "entir": [1, 2, 4, 5], "process": [1, 2, 3, 4, 5], "These": [1, 2, 4, 5], "below": [1, 2, 3, 4, 5], "hp": [1, 2, 4, 5], "1e": [1, 2, 4, 5], "batch_siz": [1, 2, 4, 5], "16": [1, 2, 4, 5], "5": [1, 2, 4, 5], "load": [1, 2], "prepar": [1, 2, 4, 5], "messi": [1, 2], "difficult": [1, 2], "maintain": [1, 2], "provid": [1, 2], "tool": [1, 2, 4, 5], "eas": [1, 2, 4, 5], "effort": [1, 2], "decoupl": [1, 2], "portion": [1, 2], "your": [1, 2, 4, 5], "pipelin": [1, 2, 4, 5], "summar": [1, 2], "three": [1, 2, 4, 5], "highlight": [1, 2, 4, 5], "demo": [1, 2, 4, 5], "store": [1, 2, 4, 5], "correspond": [1, 2, 3, 4, 5], "perform": [1, 2, 3], "manipul": [1, 2, 4, 5], "make": [1, 2, 3, 4, 5], "suitabl": [1, 2, 4, 5], "around": [1, 2, 3, 4, 5], "access": [1, 2, 4, 5], "sampl": [1, 2, 3, 4, 5], "first": [1, 2, 3, 4, 5], "cp": [1, 2, 4, 5], "r": [1, 2, 3, 4, 5], "scratch1": [1, 2, 4, 5], "07980": [1, 2, 4, 5], "sli4": [1, 2, 4, 5], "cnn_cours": [1, 2, 4, 5], "tar": [1, 2, 4, 5], "gz": [1, 2, 4, 5], "zxf": [1, 2, 4, 5], "c": [1, 2, 3, 4, 5], "l": [1, 2, 3, 4, 5], "dataset_2": [1, 2, 4, 5], "rm": [1, 2, 4, 5], "valid": [1, 2, 4, 5], "next": [1, 2, 4, 5], "path": [1, 2, 4, 5], "out": [1, 2], "base": [1, 2, 3, 4, 5], "structur": [1, 2, 3, 4, 5], "train_path": [1, 2, 4, 5], "val_path": [1, 2, 4, 5], "test_path": [1, 2, 4, 5], "none": [1, 2, 4, 5], "up": [1, 2, 3], "subsequ": [1, 2], "exist": [1, 2, 4, 5], "within": [1, 2], "imagefold": [1, 2, 4, 5], "gener": [1, 2, 3, 4, 5], "where": [1, 2, 3, 4, 5], "path_to_data": [1, 2], "png": [1, 2], "note": [1, 2, 4, 5], "directori": [1, 2, 4, 5], "name": [1, 2, 4, 5], "class": [1, 2, 3, 4, 5], "case": [1, 2], "refer": [1, 2], "level": [1, 2, 4, 5], "anoth": [1, 2, 3], "noteworthi": [1, 2], "featur": [1, 2, 3], "immedi": [1, 2], "memori": [1, 2], "thei": [1, 2], "one": [1, 2, 3, 4, 5], "compos": [1, 2, 4, 5], "appli": [1, 2, 3, 4, 5], "seri": [1, 2, 4, 5], "two": [1, 2, 3, 4, 5], "resiz": [1, 2, 4, 5], "which": [1, 2, 3, 4, 5], "specifi": [1, 2, 4, 5], "dimens": [1, 2, 4, 5], "transofrm": [1, 2, 4, 5], "totensor": [1, 2, 4, 5], "convert": [1, 2, 4, 5], "pil": [1, 2, 4, 5], "numpi": [1, 2, 4, 5], "arrrai": [1, 2], "tensor": [1, 2, 4, 5], "test": [1, 2, 4, 5], "describ": [1, 2, 4, 5], "abov": [1, 2, 4, 5], "def": [1, 2, 4, 5], "load_dataset": [1, 2, 4, 5], "img_transform": [1, 2, 4, 5], "train_dataset": [1, 2, 4, 5], "val_dataset": [1, 2, 4, 5], "test_dataset": [1, 2, 4, 5], "els": [1, 2, 4, 5], "print": [1, 2, 4, 5], "f": [1, 2, 3, 4, 5], "len": [1, 2, 4, 5], "return": [1, 2, 4, 5], "train_set": [1, 2, 4, 5], "val_set": [1, 2, 4, 5], "test_set": [1, 2, 4, 5], "1322": [1, 2, 4, 5], "363": [1, 2, 4, 5], "As": [1, 2, 3, 4, 5], "mention": [1, 2, 3], "typic": [1, 2, 4, 5], "descent": [1, 2, 3], "algorithm": [1, 2, 3], "thu": [1, 2, 3], "random": [1, 2, 3], "step": [1, 2, 3, 4, 5], "an": [1, 2, 3, 4, 5], "automat": [1, 2, 4, 5], "via": [1, 2, 3, 4, 5], "simpl": [1, 2, 3], "api": [1, 2], "instanti": [1, 2, 4, 5], "construct_dataload": [1, 2, 4, 5], "shuffl": [1, 2, 4, 5], "true": [1, 2, 3, 4, 5], "instant": 1, "train_dataload": [1, 2, 4, 5], "val_dataload": [1, 2, 4, 5], "util": [1, 2, 4, 5], "test_dataload": [1, 2, 4, 5], "befor": [1, 2, 3, 4, 5], "architectur": [1, 2, 4, 5], "some": [1, 2, 3, 4, 5], "fig": [1, 2, 4, 5], "ax": [1, 2, 4, 5], "subplot": [1, 2, 4, 5], "3": [1, 2, 3, 4, 5], "figsiz": [1, 2, 4, 5], "8": [1, 2, 4, 5], "label_map": [1, 2, 4, 5], "0": [1, 2, 3, 4, 5], "ravel": [1, 2, 4, 5], "sample_idx": [1, 2, 4, 5], "randint": [1, 2, 4, 5], "item": [1, 2, 4, 5], "img": [1, 2, 4, 5], "imshow": [1, 2, 4, 5], "permut": [1, 2, 4, 5], "reshap": [1, 2, 4, 5], "244": [1, 2, 4, 5], "set_titl": [1, 2, 4, 5], "tight_layout": [1, 2, 4, 5], "see": [1, 2, 4, 5], "contain": [1, 2, 3, 4, 5], "hous": [1, 2, 4, 5], "post": [1, 2, 3, 4, 5], "ha": [1, 2, 3, 4, 5], "been": [1, 2, 3, 4, 5], "have": [1, 2, 3, 4, 5], "classfier": [1, 2], "It": [1, 2, 3, 4, 5], "rare": [1, 2], "peopl": [1, 2], "scratch": [1, 2], "leverag": [1, 2, 4, 5], "wa": [1, 2, 4, 5], "veri": [1, 2, 3], "larg": [1, 2, 3, 4, 5], "knowledg": [1, 2], "done": [1, 2, 4, 5], "resnet18": [1, 2], "imagenet": [1, 2, 4, 5], "pretrain": [1, 2], "weight": [1, 2, 3, 4, 5], "imagenet1k_v1": [1, 2, 4, 5], "With": [1, 2], "common": [1, 2, 3], "wai": [1, 2, 4, 5], "previous": [1, 2, 4, 5], "instead": [1, 2], "acceler": [1, 2, 4, 5], "fix": [1, 2, 4, 5], "extractor": [1, 2], "That": [1, 2, 4, 5], "freez": [1, 2], "except": [1, 2, 4, 5], "last": [1, 2], "fulli": [1, 2, 3, 4, 5], "connect": [1, 2, 4, 5], "layer": [1, 2, 4, 5], "param": [1, 2], "requires_grad": [1, 2], "fals": [1, 2, 4, 5], "want": [1, 2], "fine": [1, 2], "tune": [1, 2], "could": [1, 2, 4, 5], "skip": [1, 2], "add": [1, 2, 3], "new": [1, 2, 3, 4, 5], "final": [1, 2, 4, 5], "default": [1, 2], "frozen": [1, 2], "replac": [1, 2, 4, 5], "fc": [1, 2, 4, 5], "linear": [1, 2, 3, 4, 5], "in_featur": [1, 2, 4, 5], "512": [1, 2], "out_featur": [1, 2], "1000": [1, 2], "bia": [1, 2], "input": [1, 2, 3, 4, 5], "now": [1, 2, 4, 5], "built": [1, 2, 4, 5], "readi": [1, 2, 4, 5], "To": [1, 2, 3, 4, 5], "cross": [1, 2, 3, 4, 5], "entropi": [1, 2, 3, 4, 5], "adam": [1, 2, 4, 5], "opt": [1, 2, 4, 5], "loss_fn": [1, 2, 4, 5], "crossentropyloss": [1, 2, 4, 5], "top": [1, 2], "0001": [1, 2, 5], "standard": [1, 2, 4, 5], "multipl": [1, 2, 4, 5], "given": [1, 2, 3, 4, 5], "accuraci": [1, 2, 3, 4, 5], "comput": [1, 2, 3, 4, 5], "everi": [1, 2, 3, 4, 5], "no_grad": [1, 2, 4, 5], "eval_model": [1, 2, 4, 5], "data_load": [1, 2, 4, 5], "eval": [1, 2, 4, 5], "n": [1, 2, 3, 4, 5], "enumer": [1, 2, 4, 5], "x": [1, 2, 3, 4, 5], "y": [1, 2, 3, 4, 5], "pred": [1, 2, 4, 5], "pred_label": [1, 2, 4, 5], "argmax": [1, 2, 4, 5], "axi": [1, 2, 4, 5], "sum": [1, 2, 3, 4, 5], "train_load": [1, 2, 4, 5], "val_load": [1, 2, 4, 5], "rang": [1, 2, 4, 5], "count": [1, 2], "avg_loss": [1, 2, 4, 5], "avg_acc": [1, 2, 4, 5], "start_tim": [1, 2, 4, 5], "predict": [1, 2, 3, 4, 5], "backpropog": [1, 2], "reset": [1, 2], "calcul": [1, 2, 3, 4, 5], "end_tim": [1, 2, 4, 5], "second": [1, 2, 4, 5], "averag": [1, 2, 4, 5], "val_loss": [1, 2, 4, 5], "val_acc": [1, 2, 4, 5], "val": [1, 2, 4, 5], "avail": [1, 2, 4, 5], "If": [1, 2, 3, 4, 5], "pass": [1, 2, 3, 4, 5], "cuda": [1, 2, 4, 5], "is_avail": [1, 2, 4, 5], "cpu": [1, 2, 4, 5], "type": [1, 2, 4, 5], "index": [1, 2, 4, 5], "task": [1, 2, 3, 4, 5], "monitor": [1, 2, 4, 5], "chang": [1, 2, 4, 5], "along": [1, 2], "v": [1, 2, 4, 5], "227": [1, 2], "9787924289703369": 1, "5275601744651794": 1, "05397671461105347": 1, "6563735604286194": [1, 2], "225": [1, 2], "8186715841293335": 1, "6528614163398743": 1, "04819485917687416": 1, "6944169998168945": 1, "226": [1, 2], "7649137377738953": 1, "6733433604240417": 1, "04729057475924492": 1, "6916996240615845": 1, "228": 1, "7149023413658142": 1, "6850903034210205": 1, "046091992408037186": 1, "6931818127632141": 1, "231": 1, "6881916522979736": 1, "7182228565216064": 1, "043412353843450546": 1, "7228261232376099": 1, "try": [1, 2, 4, 5], "compar": [1, 2, 4, 5], "speed": [1, 2, 4, 5], "best": [1, 2], "introduc": [1, 2, 3, 4, 5], "There": [1, 2, 3, 4, 5], "were": [1, 2], "major": [1, 2, 4, 5], "sever": [1, 2, 4, 5], "modif": [1, 2, 3, 4, 5], "workflow": [1, 2], "improv": [1, 2, 4, 5], "http": [2, 3, 5], "org": [2, 5], "f37072fd": 2, "pth": [2, 5], "checkpoint": 2, "100": [2, 5], "44": 2, "7m": 2, "00": [2, 5], "103mb": 2, "num_ftr": [2, 4, 5], "sequenti": [2, 4, 5], "relu": [2, 3, 4, 5], "zero_grad": [2, 4, 5], "backward": [2, 4, 5], "970403254032135": 2, "5445783138275146": 2, "05382955074310303": 2, "222": 2, "7938651442527771": 2, "6698794960975647": 2, "04907182604074478": 2, "6699604988098145": 2, "223": 2, "7483504414558411": 2, "6772590279579163": 2, "04733126237988472": 2, "7106617093086243": 2, "6864457130432129": 2, "044527120888233185": 2, "7188735008239746": 2, "6822027564048767": 2, "7106927633285522": 2, "04329155385494232": 2, "7119565606117249": 2, "05": [2, 4, 5], "806357383728027": 2, "0007530120201408863": 2, "5797683596611023": 2, "817122459411621": 2, "5735561847686768": 2, "224": 2, "80456256866455": 2, "5777430534362793": 2, "217": 2, "827596664428711": 2, "5745983123779297": 2, "218": 2, "808821678161621": 2, "5743387341499329": 2, "001": 2, "814985275268555": 2, "5782486796379089": 2, "809774398803711": 2, "571698009967804": 2, "801504135131836": 2, "5749053359031677": 2, "800519943237305": 2, "0019578312058001757": 2, "5774373412132263": 2, "804729461669922": 2, "5809144377708435": 2, "ml": 3, "dl": 3, "model": 3, "includ": 3, "multilay": 3, "convolit": 3, "also": [3, 4, 5], "introduct": 3, "dataload": 3, "do": 3, "transform": 3, "hand": 3, "session": 3, "natur": 3, "hazard": 3, "detect": [3, 4, 5], "exampl": [3, 4, 5], "consist": 3, "extract": 3, "encod": [3, 4, 5], "establish": 3, "between": 3, "decod": 3, "sourc": 3, "cen": 3, "approach": 3, "identifi": [3, 4, 5], "solv": 3, "problem": 3, "contrast": 3, "present": 3, "mah": 3, "about": 3, "concept": 3, "sinc": 3, "origin": [3, 4, 5], "assumpt": 3, "behind": 3, "percept": 3, "dataset": 3, "linearli": 3, "separ": 3, "e": [3, 4, 5], "exit": 3, "hyperplan": 3, "perfectli": 3, "divid": 3, "small": 3, "group": 3, "suppos": 3, "find": [3, 4, 5], "seper": 3, "clases": 3, "finit": 3, "held": 3, "fail": 3, "defin": 3, "h": 3, "dot": 3, "w": 3, "b": 3, "posit": 3, "neg": 3, "wei": 3, "consid": 3, "build": 3, "piecewis": 3, "function": 3, "simul": 3, "target": 3, "curv": 3, "call": [3, 4, 5], "neuron": 3, "its": 3, "figur": [3, 4, 5], "show": [3, 4, 5], "binari": 3, "li": 3, "formular": 3, "shown": 3, "matrx": 3, "o": [3, 4, 5], "output": 3, "vector": 3, "scaler": 3, "sigma": 3, "activ": 3, "g": 3, "sigmoid": 3, "tx": 3, "For": [3, 4, 5], "onli": [3, 4, 5], "possibl": 3, "other": 3, "sign": 3, "\ud835\udc66": 3, "mathbb": 3, "multiclass": 3, "per": 3, "op": 3, "observ": 3, "evalu": 3, "NOT": 3, "either": [3, 4, 5], "assum": [3, 4, 5], "same": [3, 4, 5], "produc": [3, 4, 5], "A": [3, 4, 5], "idea": 3, "minim": 3, "loss": 3, "numer": [3, 4, 5], "optim": 3, "like": [3, 4, 5], "commonli": 3, "valu": 3, "At": 3, "tweak": 3, "math": 3, "equat": [3, 4, 5], "w_t": 3, "frac": [3, 4, 5], "sum_": 3, "nl": 3, "x_i": 3, "y_i": 3, "w_": 3, "t": 3, "alpha": [3, 4, 5], "nabla": 3, "backpropag": 3, "fortun": 3, "manual": 3, "deriv": 3, "differ": 3, "framework": 3, "tensorflow": 3, "auto": 3, "engin": 3, "usual": 3, "implement": [3, 4, 5], "graph": 3, "backprop": 3, "zha": 3, "oper": [3, 4, 5], "conver": 3, "measur": 3, "variat": 3, "probabl": [3, 4, 5], "distribut": 3, "definit": 3, "p_i": 3, "exp": 3, "j": 3, "cexp": 3, "x_j": 3, "ct_ilog": 3, "t_i": 3, "highest": [3, 4, 5], "happen": 3, "cannot": 3, "reflect": 3, "relat": 3, "poorli": 3, "workd": 3, "more": [3, 4, 5], "complex": 3, "solut": 3, "ad": [3, 4, 5], "boost": 3, "etc": [3, 4, 5], "justifi": 3, "lead": [3, 4, 5], "poor": 3, "less": 3, "bag": 3, "jog": 3, "mnist": 3, "handwritten": 3, "digit": 3, "28": 3, "pixel": [3, 4, 5], "width": 3, "height": 3, "greyscal": 3, "channel": 3, "howev": [3, 4, 5], "recognit": 3, "so": [3, 4, 5], "good": 3, "main": [3, 4, 5], "reason": 3, "spatial": 3, "inform": [3, 4, 5], "object": [3, 4, 5], "particular": [3, 4, 5], "pattern": 3, "gather": 3, "move": 3, "place": [3, 4, 5], "should": [3, 4, 5], "still": 3, "abl": 3, "recogn": 3, "even": 3, "might": 3, "seen": 3, "vid": 3, "core": 3, "block": 3, "consum": 3, "map": 3, "filter": 3, "product": 3, "fed": 3, "ensur": [3, 4, 5], "invari": 3, "properti": 3, "No": 3, "matter": 3, "correctli": 3, "ber": 3, "bri": 3, "exploit": 3, "rotat": [3, 4, 5], "shift": 3, "One": 3, "cope": 3, "issu": 3, "element": 3, "preserv": 3, "max": 3, "min": [3, 5], "avg": 3, "reduc": 3, "randomli": [3, 4, 5], "drop": 3, "increas": [3, 4, 5], "situat": 3, "proven": 3, "effect": 3, "regular": [3, 4, 5], "thing": 3, "dure": [3, 4, 5], "stage": 3, "select": [3, 4, 5], "contribut": 3, "wors": 3, "caus": 3, "higher": 3, "error": 3, "explan": 3, "vanish": 3, "preced": 3, "help": [3, 4, 5], "densenet": 3, "signal": 3, "achiev": 3, "smaller": 3, "hlvdmw17": 3, "collect": 3, "just": 3, "them": 3, "instanc": 3, "horizont": 3, "vertic": 3, "flip": 3, "clip": 3, "u": 3, "larger": [3, 4, 5], "better": 3, "offer": 3, "doe": [3, 4, 5], "while": 3, "enough": 3, "work": 3, "again": [3, 4, 5], "adapt": 3, "fewer": 3, "bernard": 3, "url": 3, "cornel": 3, "edu": 3, "brows": 3, "roster": 3, "fa22": 3, "ec": 3, "5775": 3, "britz": 3, "dennybritz": 3, "com": 3, "wildml": 3, "understand": 3, "nlp": 3, "centin": 3, "nheri": 3, "simcent": 3, "simcenter_designsafe_ml_2022": 3, "hassan": 3, "neuroh": 3, "io": 3, "en": 3, "vgg16": 3, "gao": 3, "huang": 3, "zhuang": 3, "liu": 3, "lauren": 3, "van": 3, "der": 3, "maaten": 3, "kilian": 3, "q": 3, "weinberg": 3, "dens": 3, "proceed": 3, "ieee": 3, "confer": 3, "vision": 3, "4700": 3, "4708": 3, "joglekar": 3, "srjoglekar246": 3, "human": 3, "behavior": 3, "5186df1e7d19": 3, "cs231n": 3, "github": 3, "mahapatra": 3, "towardsdatasci": 3, "why": 3, "1b6a99177063": 3, "opennn": 3, "www": 3, "net": 3, "vidhya": 3, "analyt": 3, "8d0a292b4498": 3, "cs4780": 3, "2023fa": 3, "lectur": 3, "lecturenote03": 3, "html": 3, "zhang": 3, "expand": [4, 5], "advanc": [4, 5], "techniqu": [4, 5], "part1": [4, 5], "addit": [4, 5], "previou": [4, 5], "arrai": [4, 5], "both": [4, 5], "preprocess": [4, 5], "format": [4, 5], "dimension": [4, 5], "divers": [4, 5], "translat": [4, 5], "crop": [4, 5], "autoaug": [4, 5], "varieti": [4, 5], "procedur": [4, 5], "search": [4, 5], "polici": [4, 5], "val_img_transform": [4, 5], "train_img_transform": [4, 5], "constructor": [4, 5], "strategi": [4, 5], "modifi": [4, 5], "consolid": [4, 5], "code": [4, 5], "creat": [4, 5], "unlik": [4, 5], "resnet34": [4, 5], "getresnet": [4, 5], "conv": [4, 5], "conv_param": [4, 5], "require_grad": [4, 5], "mlp": [4, 5], "made": [4, 5], "hot": [4, 5], "represent": [4, 5], "integ": [4, 5], "cat": [4, 5], "versu": [4, 5], "dog": [4, 5], "repres": [4, 5], "would": [4, 5], "most": [4, 5], "outcom": [4, 5], "6": [4, 5], "mean": [4, 5], "60": [4, 5], "chanc": [4, 5], "introdcu": [4, 5], "nois": [4, 5], "alter": [4, 5], "decreas": [4, 5], "confid": [4, 5], "label_smooth": [4, 5], "control": [4, 5], "label_": [4, 5], "soft": [4, 5], "hard": [4, 5], "nc": [4, 5], "being": [4, 5], "times0": [4, 5], "9": [4, 5], "95": [4, 5], "plai": [4, 5], "role": [4, 5], "obtain": [4, 5], "often": [4, 5], "advantag": [4, 5], "lower": [4, 5], "ani": [4, 5], "gain": [4, 5], "paper": [4, 5], "those": [4, 5], "lr_schedul": [4, 5], "reducelronplateau": [4, 5], "occur": [4, 5], "factor": [4, 5], "after": [4, 5], "schedul": [4, 5], "take": [4, 5], "lot": [4, 5], "power": [4, 5], "addition": [4, 5], "mai": [4, 5], "end": [4, 5], "initi": [4, 5], "suffici": [4, 5], "underfit": [4, 5], "resum": [4, 5], "convers": [4, 5], "mani": [4, 5], "overfit": [4, 5], "scenario": [4, 5], "mechan": [4, 5], "save": [4, 5], "lose": [4, 5], "rest": [4, 5], "far": [4, 5], "load_checkpoint": [4, 5], "version": [4, 5], "checkpoint_path": [4, 5], "map_loc": [4, 5], "alreadi": [4, 5], "file": [4, 5], "intend": [4, 5], "model_folder_path": [4, 5], "getcwd": [4, 5], "output_model": [4, 5], "makedir": [4, 5], "exist_ok": [4, 5], "filenam": [4, 5], "checkpoint_fil": [4, 5], "best_model": [4, 5], "pt": [4, 5], "itself": [4, 5], "experi": [4, 5], "prev_best_val_acc": [4, 5], "loop": [4, 5], "associ": [4, 5], "earlier": [4, 5], "disk": [4, 5], "best_val_acc": [4, 5], "total_tim": [4, 5], "action": [4, 5], "nprev": [4, 5], "acc": [4, 5], "cur": [4, 5], "finish": [4, 5], "metric": [4, 5], "current": [4, 5], "prev": [4, 5], "706768810749054": 4, "414": 4, "054781472485467615": 4, "6099397540092468": 4, "04778964817523956": 4, "7895256876945496": 4, "410": [4, 5], "04340849735560905": 4, "7557228207588196": 4, "04375723376870155": 4, "03606990118432476": 4, "8379518389701843": 4, "05057337507605553": 4, "73097825050354": 4, "408": [4, 5], "03369753223855093": 4, "8668674230575562": 4, "042350128293037415": 4, "7623517513275146": 4, "411": [4, 5], "029927182439939082": 4, "8981927633285522": 4, "05253813788294792": 4, "7161561250686646": 4, "without": [4, 5], "load_model_fm_checkpoint": [4, 5], "primitive_model": [4, 5], "load_state_dict": [4, 5], "model_state_dict": [4, 5], "model_dump_dir": [4, 5], "ckpt": [4, 5], "filenotfounderror": [4, 5], "pleas": [4, 5], "follo": [4, 5], "load_test_dataset": [4, 5], "cross_valid": [4, 5], "displai": [4, 5], "match": [4, 5], "choos": [4, 5], "random_idx": [4, 5], "sample_imag": [4, 5], "image_idx": [4, 5], "unsqueez": [4, 5], "actual": [4, 5], "213": [4, 5], "b627a593": 5, "83": 5, "3m": 5, "111mb": 5, "begin": 5, "align": 5, "mode": 5, "patienc": 5, "min_lr": 5, "verbos": 5, "get_last_lr": 5, "state_dict": 5, "698616623878479": 5, "055991143006158164": 5, "6024096012115479": 5, "04682241007685661": 5, "7638339996337891": 5, "04354576942881187": 5, "7448794841766357": 5, "04492916166782379": 5, "780138373374939": 5, "409": 5, "0374780154057655": 5, "8299698233604431": 5, "044687770307064056": 5, "405": 5, "03239582747849355": 5, "8787650465965271": 5, "050802767276763916": 5, "728507936000824": 5, "031860928318227635": 5, "8873493671417236": 5, "04592260718345642": 5, "7732213139533997": 5, "288": 5, "0487391661224236": 5, "6658132076263428": 5, "04293294996023178": 5, "6716897487640381": 5, "289": 5, "019708584861672786": 5, "8941264748573303": 5, "03811215981841087": 5, "7734683752059937": 5, "010429383531852389": 5, "9403614401817322": 5, "04305000975728035": 5, "7475296854972839": 5, "008909780737601715": 5, "955572247505188": 5, "048485103994607925": 5, "7351778149604797": 5, "008085134100163333": 5, "9498493671417236": 5, "05932692065834999": 5, "7324604392051697": 5}, "objects": {}, "objtypes": {}, "objnames": {}, "titleterms": {"build": [0, 1, 2, 4, 5], "scalabl": 0, "cnn": [0, 1, 2, 3, 4, 5], "model": [0, 1, 2, 4, 5], "classifi": [1, 2, 4, 5], "pytorch": [1, 2, 4, 5], "part": [1, 2, 4, 5], "1": [1, 2], "dataset": [1, 2, 4, 5], "loader": [1, 2, 4, 5], "transform": [1, 2, 4, 5], "download": [1, 2, 4, 5], "construct": [1, 2, 4, 5], "dataload": [1, 2, 4, 5], "visual": [1, 2, 4, 5], "design": [1, 2, 4, 5], "safe": [1, 2, 4, 5], "neural": [1, 2, 3, 4, 5], "network": [1, 2, 3, 4, 5], "resnet": [1, 2, 4, 5], "transfer": [1, 2, 3, 4, 5], "learn": [1, 2, 3, 4, 5], "train": [1, 2, 3, 4, 5], "defin": [1, 2, 4, 5], "loss": [1, 2, 4, 5], "function": [1, 2, 4, 5], "optim": [1, 2, 4, 5], "evalu": [1, 2, 4, 5], "check": [1, 2, 4, 5], "gpu": [1, 2, 4, 5], "move": [1, 2, 4, 5], "correct": [1, 2, 4, 5], "devic": [1, 2, 4, 5], "option": [1, 2, 4, 5], "exercis": [1, 2, 4, 5], "conclus": [1, 2], "imag": [3, 4, 5], "classif": 3, "convolut": 3, "outlin": 3, "tradit": 3, "method": 3, "what": 3, "i": 3, "deep": 3, "multi": 3, "layer": 3, "perceptron": 3, "mlp": 3, "test": 3, "softmax": 3, "regress": 3, "underfit": 3, "v": 3, "overfit": 3, "from": 3, "pool": 3, "dropout": 3, "residu": 3, "connect": 3, "data": 3, "augment": [3, 4, 5], "refer": 3, "2": [4, 5], "label": [4, 5], "smooth": [4, 5], "reduc": [4, 5], "rate": [4, 5], "plateau": [4, 5], "set": [4, 5], "up": [4, 5], "checkpoint": [4, 5], "we": [4, 5], "read": [4, 5], "back": [4, 5], "best": [4, 5], "explor": [4, 5], "perform": [4, 5], "variou": [4, 5], "load": [4, 5], "infer": [4, 5], "random": [4, 5]}, "envversion": {"sphinx.domains.c": 3, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 9, "sphinx.domains.index": 1, "sphinx.domains.javascript": 3, "sphinx.domains.math": 2, "sphinx.domains.python": 4, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.intersphinx": 1, "sphinxcontrib.bibtex": 9, "sphinx": 60}, "alltitles": {"Building Scalable CNN models": [[0, "building-scalable-cnn-models"]], "Building a CNN Classifier with PyTorch: Part 1": [[1, "building-a-cnn-classifier-with-pytorch-part-1"], [2, "building-a-cnn-classifier-with-pytorch-part-1"]], "Dataset Loaders and Transforms": [[1, "dataset-loaders-and-transforms"], [2, "dataset-loaders-and-transforms"], [4, "dataset-loaders-and-transforms"], [5, "dataset-loaders-and-transforms"]], "Downloading dataset": [[1, "downloading-dataset"], [2, "downloading-dataset"], [4, "downloading-dataset"], [5, "downloading-dataset"]], "Dataset and Transforms": [[1, "dataset-and-transforms"], [2, "dataset-and-transforms"]], "Dataset": [[1, "dataset"], [2, "dataset"]], "Transforms": [[1, "transforms"], [2, "transforms"], [4, "transforms"], [5, "transforms"]], "Construct Dataloaders": [[1, "construct-dataloaders"], [2, "construct-dataloaders"], [4, "construct-dataloaders"], [5, "construct-dataloaders"]], "Visualizing the Design Safe Dataset": [[1, "visualizing-the-design-safe-dataset"], [2, "visualizing-the-design-safe-dataset"]], "Building the Neural Network": [[1, "building-the-neural-network"], [2, "building-the-neural-network"], [4, "building-the-neural-network"], [5, "building-the-neural-network"]], "ResNet": [[1, "resnet"], [2, "resnet"]], "Transfer Learning": [[1, "transfer-learning"], [2, "transfer-learning"], [3, "transfer-learning"]], "Training the Neural Network": [[1, "training-the-neural-network"], [2, "training-the-neural-network"], [4, "training-the-neural-network"], [5, "training-the-neural-network"]], "Define Loss Function and Optimizer": [[1, "define-loss-function-and-optimizer"], [2, "define-loss-function-and-optimizer"], [4, "define-loss-function-and-optimizer"], [5, "define-loss-function-and-optimizer"]], "Train and Model Evaluation Functions": [[1, "train-and-model-evaluation-functions"], [2, "train-and-model-evaluation-functions"], [4, "train-and-model-evaluation-functions"], [5, "train-and-model-evaluation-functions"]], "Check for GPU and move model to correct device": [[1, "check-for-gpu-and-move-model-to-correct-device"], [2, "check-for-gpu-and-move-model-to-correct-device"], [4, "check-for-gpu-and-move-model-to-correct-device"], [5, "check-for-gpu-and-move-model-to-correct-device"]], "Train Model": [[1, "train-model"], [2, "train-model"], [4, "train-model"], [5, "train-model"]], "OPTIONAL EXERCISE": [[1, "optional-exercise"], [2, "optional-exercise"], [4, "optional-exercise"], [5, "optional-exercise"]], "Conclusion": [[1, "conclusion"], [2, "conclusion"]], "Image Classification and Convolutional Neural Network": [[3, "image-classification-and-convolutional-neural-network"]], "Outline": [[3, "outline"]], "Traditional Methods": [[3, "traditional-methods"]], "What is Deep Learning?": [[3, "what-is-deep-learning"]], "Multi-Layer Perceptron (MLP)": [[3, "multi-layer-perceptron-mlp"], [3, "id5"]], "What is Perceptron": [[3, "what-is-perceptron"]], "Training and Testing": [[3, "training-and-testing"]], "Softmax Regression": [[3, "softmax-regression"]], "Underfitting vs. Overfitting": [[3, "underfitting-vs-overfitting"]], "Convolutional Neural Network (CNN)": [[3, "convolutional-neural-network-cnn"]], "From MLP to CNN": [[3, "from-mlp-to-cnn"]], "Convolutional Layer": [[3, "convolutional-layer"]], "Pooling Layer": [[3, "pooling-layer"]], "Dropout layer": [[3, "dropout-layer"]], "Residual connection": [[3, "residual-connection"]], "Data augmentation": [[3, "data-augmentation"]], "Reference": [[3, "reference"]], "Building a CNN Classifier with PyTorch: Part 2": [[4, "building-a-cnn-classifier-with-pytorch-part-2"], [5, "building-a-cnn-classifier-with-pytorch-part-2"]], "Visualizing the Augmented Design Safe Dataset": [[4, "visualizing-the-augmented-design-safe-dataset"], [5, "visualizing-the-augmented-design-safe-dataset"]], "ResNet and Transfer Learning": [[4, "resnet-and-transfer-learning"], [5, "resnet-and-transfer-learning"]], "Label smoothing": [[4, "label-smoothing"], [5, "label-smoothing"]], "Reduced learning rate on plateau": [[4, "reduced-learning-rate-on-plateau"], [5, "reduced-learning-rate-on-plateau"]], "Setting up Checkpoints": [[4, "setting-up-checkpoints"], [5, "setting-up-checkpoints"]], "We read back the best model and explore performance on various images": [[4, "we-read-back-the-best-model-and-explore-performance-on-various-images"], [5, "we-read-back-the-best-model-and-explore-performance-on-various-images"]], "Read the model checkpoint": [[4, "read-the-model-checkpoint"], [5, "read-the-model-checkpoint"]], "Load in the dataset": [[4, "load-in-the-dataset"], [5, "load-in-the-dataset"]], "Perform inference on a random image": [[4, "perform-inference-on-a-random-image"], [5, "perform-inference-on-a-random-image"]]}, "indexentries": {}})