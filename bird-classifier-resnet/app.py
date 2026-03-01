import gradio as gr
from fastai.vision.all import load_learner, PILImage

try:
    learn = load_learner('bird_classifier_model.pkl')
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")

def predict_image(img):
    fastai_img = PILImage.create(img)
    pred, pred_idx, probs = learn.predict(fastai_img)
    return {learn.dls.vocab[i]: float(probs[i]) for i in range(len(learn.dls.vocab))}
interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(num_top_classes=2),
    title="Bird vs. Forest Classifier",
    description="Upload an image to test the ResNet18 transfer learning inference engine."
)

interface.launch()