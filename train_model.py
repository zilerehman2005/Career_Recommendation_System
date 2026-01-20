"""
Train ensemble model and save for Streamlit app
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from imblearn.over_sampling import SMOTE
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# ─── 1. Load & Preprocess ────────────────────────────────────────
print("="*70)
print("🚀 AI Career Recommendation - Ensemble Model Training")
print("="*70)
print("\n📂 Loading data...")
df = pd.read_csv("AI_Career_Recommendation_8000.csv")
print(f"✓ Loaded {len(df)} records")

# Drop unnecessary columns
df = df.drop(columns=['CandidateID', 'Name', 'Recommendation_Score'], errors='ignore')

# Prepare target variable
y = df['Recommended_Career']
X = df.drop(columns=['Recommended_Career'])

career_classes = sorted(y.unique())
class_to_idx = {c: i for i, c in enumerate(career_classes)}
y_encoded = y.map(class_to_idx).values
num_classes = len(career_classes)

print(f"✓ Found {num_classes} career classes")
print(f"✓ Careers: {', '.join(career_classes[:5])}..." if len(career_classes) > 5 else f"✓ Careers: {', '.join(career_classes)}")

# ─── 2. Feature Engineering ─────────────────────────────────────
print("\n🔧 Engineering features...")

# Age
X['Age'] = pd.to_numeric(X['Age'], errors='coerce')

# Education one-hot encoding
education_cols_before = list(X.columns)
X = pd.get_dummies(X, columns=['Education'], prefix='Edu')
education_cols = [col for col in X.columns if col.startswith('Edu_') and col not in education_cols_before]

# Skills one-hot encoding
skills_col_before = list(X.columns)
X['Skills'] = X['Skills'].fillna('')
skills_set = set()
for row in X['Skills']:
    skills_set.update(s.strip() for s in row.split(';') if s.strip())
for skill in skills_set:
    X[f'Skills_{skill}'] = X['Skills'].apply(lambda x: 1 if skill in [s.strip() for s in x.split(';')] else 0)
X = X.drop(columns=['Skills'])
skill_cols = [col for col in X.columns if col.startswith('Skills_') and col not in skills_col_before]

# Interests one-hot encoding
interests_col_before = list(X.columns)
X['Interests'] = X['Interests'].fillna('')
interests_set = set()
for row in X['Interests']:
    interests_set.update(s.strip() for s in row.split(';') if s.strip())
for interest in interests_set:
    X[f'Interests_{interest}'] = X['Interests'].apply(lambda x: 1 if interest in [s.strip() for s in x.split(';')] else 0)
X = X.drop(columns=['Interests'])
interest_cols = [col for col in X.columns if col.startswith('Interests_') and col not in interests_col_before]

print(f"✓ Total features: {X.shape[1]}")
print(f"  - Education categories: {len(education_cols)}")
print(f"  - Skills: {len(skill_cols)}")
print(f"  - Interests: {len(interest_cols)}")

# ─── 3. Train-Test Split ────────────────────────────────────────
print("\n📊 Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
)
print(f"✓ Training set: {len(X_train)} samples")
print(f"✓ Test set: {len(X_test)} samples")

# ─── 4. Scale Features ──────────────────────────────────────────
print("\n⚖️  Scaling features...")
scaler = StandardScaler()
X_train[['Age']] = scaler.fit_transform(X_train[['Age']])
X_test[['Age']] = scaler.transform(X_test[['Age']])

X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

# ─── 5. Apply SMOTE ─────────────────────────────────────────────
print("\n🔄 Applying SMOTE for class balance...")
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(f"✓ Balanced training set: {len(X_train_sm)} samples")

# ─── 6. Create DataLoaders ──────────────────────────────────────
train_ds = TensorDataset(
    torch.from_numpy(X_train_sm.values), 
    torch.from_numpy(y_train_sm).long()
)
test_ds = TensorDataset(
    torch.from_numpy(X_test.values), 
    torch.from_numpy(y_test).long()
)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

# ─── 7. MLP Model Definition ────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),         nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),          nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.net(x)

# ─── 8. Train One Model ─────────────────────────────────────────
def train_one_model(seed, model_idx):
    torch.manual_seed(seed)
    model = MLP(X.shape[1], num_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    losses = []
    print(f"\n🤖 Training Model {model_idx}/5 (seed={seed})")
    print("-" * 50)
    
    for epoch in range(1, 51):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        
        if epoch % 10 == 0:
            print(f"  Epoch {epoch:2d}/50  Loss: {avg_loss:.4f}")
    
    return model, losses

# ─── 9. Train Ensemble ──────────────────────────────────────────
print("\n" + "="*70)
print("🎯 Training Ensemble (5 Models)")
print("="*70)

models = []
all_losses = []
seeds = [42, 123, 777, 999, 2025]

for i, seed in enumerate(seeds, 1):
    model, losses = train_one_model(seed, i)
    models.append(model)
    all_losses.append(losses)

# ─── 10. Save Models ────────────────────────────────────────────
print("\n💾 Saving models...")
os.makedirs('models', exist_ok=True)

# Save each model
for i, model in enumerate(models):
    torch.save(model.state_dict(), f'models/model_{i}.pth')
print(f"✓ Saved {len(models)} model files")

# Save metadata
metadata = {
    'num_features': X.shape[1],
    'num_classes': num_classes,
    'career_classes': career_classes,
    'class_to_idx': class_to_idx,
    'feature_names': list(X.columns),
    'education_columns': education_cols,
    'skill_columns': skill_cols,
    'interest_columns': interest_cols,
    'scaler': scaler
}

with open('models/model_metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)
print("✓ Saved model metadata")

# ─── 11. Evaluate Ensemble ──────────────────────────────────────
print("\n📈 Evaluating ensemble on test set...")

def ensemble_predict(loader):
    probs = []
    for model in models:
        model.eval()
        model_probs = []
        with torch.no_grad():
            for xb, _ in loader:
                out = torch.softmax(model(xb), dim=1)
                model_probs.append(out.cpu().numpy())
        probs.append(np.concatenate(model_probs))
    
    # Average probabilities
    ensemble_prob = np.mean(probs, axis=0)
    return np.argmax(ensemble_prob, axis=1)

y_pred = ensemble_predict(test_loader)

# Calculate metrics
acc = accuracy_score(y_test, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(
    y_test, y_pred, average='macro', zero_division=0
)

# ─── 12. Display Results ────────────────────────────────────────
print("\n" + "="*70)
print("🎉 ENSEMBLE TRAINING COMPLETE!")
print("="*70)
print(f"\n📊 Test Set Performance:")
print(f"   Accuracy:  {acc*100:.2f}%")
print(f"   Precision: {prec:.4f}")
print(f"   Recall:    {rec:.4f}")
print(f"   F1-Score:  {f1:.4f}")

# Per-class performance (top 10 classes)
print(f"\n📋 Per-Class Performance (Top 10):")
print("-" * 70)
report = classification_report(y_test, y_pred, target_names=career_classes, output_dict=True, zero_division=0)
class_f1_scores = [(career, report[career]['f1-score']) for career in career_classes]
class_f1_scores.sort(key=lambda x: x[1], reverse=True)

for i, (career, f1_score) in enumerate(class_f1_scores[:10], 1):
    print(f"   {i:2d}. {career:30s} F1: {f1_score:.4f}")

# ─── 13. Plot Loss Curve ────────────────────────────────────────
print("\n📉 Creating loss curve visualization...")
avg_losses = np.mean(all_losses, axis=0)

plt.figure(figsize=(12, 6))
plt.plot(range(1, len(avg_losses)+1), avg_losses, 'b-', linewidth=2, label='Average Loss')

# Plot individual model losses (lighter)
for i, losses in enumerate(all_losses):
    plt.plot(range(1, len(losses)+1), losses, alpha=0.3, linewidth=1, label=f'Model {i+1}')

plt.title("Training Loss - 5-Model Ensemble", fontsize=16, fontweight='bold')
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Cross-Entropy Loss", fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("ensemble_loss_curve.png", dpi=150)
print("✓ Saved loss curve → ensemble_loss_curve.png")

# ─── 14. Summary ────────────────────────────────────────────────
print("\n" + "="*70)
print("✅ ALL DONE! You can now run the Streamlit app:")
print("="*70)
print("\n   streamlit run app.py\n")
print("📁 Generated files:")
print("   • models/model_0.pth through model_4.pth")
print("   • models/model_metadata.pkl")
print("   • ensemble_loss_curve.png")
print("\n" + "="*70)
