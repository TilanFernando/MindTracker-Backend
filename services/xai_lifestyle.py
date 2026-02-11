import shap
import pandas as pd
import numpy as np

# Explain which lifestyle factors influenced the result
def explain_prediction(model, input_df, class_names):
    """
    Computes SHAP values for a single input row within the scikit-learn pipeline.
    """
    try:
        preprocessor = model.named_steps.get('preprocess')
        classifier = model.named_steps.get('clf')

        if not preprocessor or not classifier:
            preprocessor = model.steps[0][1]
            classifier = model.steps[-1][1]

        X_transformed = preprocessor.transform(input_df)
        
        try:
            feature_names = preprocessor.get_feature_names_out()
        except Exception:
            feature_names = [f"Feature {i}" for i in range(X_transformed.shape[1])]

        explainer = shap.TreeExplainer(classifier)
        shap_vals = explainer.shap_values(X_transformed)

        probs = model.predict_proba(input_df)[0]
        predicted_class_idx = np.argmax(probs)

        if isinstance(shap_vals, list):
            class_impacts = shap_vals[predicted_class_idx][0]
        else:
            if len(shap_vals.shape) == 3:
                class_impacts = shap_vals[0, :, predicted_class_idx]
            else:
                class_impacts = shap_vals[0]

        impact_list = []
        for feat, impact in zip(feature_names, class_impacts):
            clean_feat = feat.split('__')[-1]
            impact_list.append({
                "feature": clean_feat,
                "impact": float(abs(impact))
            })

        return sorted(impact_list, key=lambda x: x['impact'], reverse=True)

    except Exception as e:
        print(f"SHAP explanation failed: {e}")
        return []
