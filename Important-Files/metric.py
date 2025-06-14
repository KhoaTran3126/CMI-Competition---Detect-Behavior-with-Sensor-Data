from sklearn.metrics import f1_score

class ParticipantVisibleError(Exception):
    """Errors raised here will be shown directly to the competitor."""
    pass


class CompetitionMetric:
    """Hierarchical macro F1 for the CMI 2025 challenge."""
    def __init__(self):
        self.target_gestures = [
            'Above ear - pull hair',
            'Cheek - pinch skin',
            'Eyebrow - pull hair',
            'Eyelash - pull hair',
            'Forehead - pull hairline',
            'Forehead - scratch',
            'Neck - pinch skin',
            'Neck - scratch',
        ]
        self.non_target_gestures = [
            'Write name on leg',
            'Wave hello',
            'Glasses on/off',
            'Text on phone',
            'Write name in air',
            'Feel around in tray and pull out an object',
            'Scratch knee/leg skin',
            'Pull air toward your face',
            'Drink from bottle/cup',
            'Pinch knee/leg skin'
        ]
        self.all_classes = self.target_gestures + self.non_target_gestures

    def calculate_hierarchical_f1(
        self,
        sol: pd.DataFrame,
        sub: pd.DataFrame
    ) -> float:

        # Validate gestures
        invalid_types = {i for i in sub['gesture'].unique() if i not in self.all_classes}
        if invalid_types:
            raise ParticipantVisibleError(
                f"Invalid gesture values in submission: {invalid_types}"
            )

        # Compute binary F1 (Target vs Non-Target)
        y_true_bin = sol['gesture'].isin(self.target_gestures).values
        y_pred_bin = sub['gesture'].isin(self.target_gestures).values
        
        f1_binary = f1_score(y_true_bin, y_pred_bin, pos_label=True, zero_division=0, average='binary')

        # Build multi-class labels for gestures
        y_true_mc = sol['gesture'].apply(lambda x: x if x in self.target_gestures else 'non_target')
        y_pred_mc = sub['gesture'].apply(lambda x: x if x in self.target_gestures else 'non_target')

        f1_macro = f1_score(y_true_mc, y_pred_mc, average='macro', zero_division=0)

        return f1_binary, f1_macro, (f1_binary+f1_macro)/2.0


def F1_score(y_val, y_pred, lbl_encoder, choice="weighted_score") -> float:
    """
    Provides competition's F1 score
    y_val: truth labels
    y_pred: predicted labels
    lbl_encoder: label encoder
    choice: choice of score to return. Can be "binary", "macro", "weighted_score", or None to receive all scores
    """
    metric = CompetitionMetric()
    y_val  = pd.DataFrame({'id':range(len(y_val)), 
                           'gesture':y_val})
    y_pred = pd.DataFrame({'id':range(len(y_pred)), 
                           'gesture':y_pred})

    ## Convert numeric labels to original descriptions
    y_val["gesture"]  = lbl_encoder.inverse_transform(y_val["gesture"])
    y_pred["gesture"] = lbl_encoder.inverse_transform(y_pred["gesture"])

    ## Compute scores
    binary, macro, weighted_score = metric.calculate_hierarchical_f1(y_val, y_pred)

    ## Returns result
    if choice=="binary": return binary
    elif choice=="macro": return macro
    elif choice=="weighted_score": return weighted_score
    else: return (binary, macro, weighted_score)
