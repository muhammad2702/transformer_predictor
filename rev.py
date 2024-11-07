import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import logging
from ta.volatility import AverageTrueRange, BollingerBands
from ta.momentum import RSIIndicator
from ta.trend import MACD, ADXIndicator
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    MultiHeadAttention,
    LayerNormalization,
    GlobalAveragePooling1D,
    Add
)
from tensorflow.keras.models import Model
import tensorflow as tf
from binance.client import Client


logging.getLogger("tensorflow").setLevel(logging.ERROR)

def load_data(client, symbol, interval, limit):
  
    try:
        historicals = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        data = []
        for kl in historicals:
            open_time = pd.to_datetime(kl[0], unit='ms')
            open_price = float(kl[1])
            high = float(kl[2])
            low = float(kl[3])
            close = float(kl[4])
            volume = float(kl[5])
            data.append([open_time, open_price, high, low, close, volume])
        df = pd.DataFrame(data, columns=['open_time', 'open', 'high', 'low', 'close', 'volume'])
        df.set_index('open_time', inplace=True)
        return df
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

def add_technical_indicators(df):
  
    # RSI
    rsi = RSIIndicator(close=df['close'], window=14)
    df['RSI'] = rsi.rsi()
    
    # MACD
    macd = MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_diff'] = macd.macd_diff()
    
    # ATR
    atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['ATR'] = atr.average_true_range()
    
    # Bollinger Bands Width
    bollinger = BollingerBands(close=df['close'], window=20, window_dev=2)
    df['BB_upper'] = bollinger.bollinger_hband()
    df['BB_lower'] = bollinger.bollinger_lband()
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['close']
    
    # ADX
    adx = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['ADX'] = adx.adx()
    

    df.dropna(inplace=True)
    return df

def classify_market_environments(df):
    
    # Volatility Classification
    df['ATR_mavg'] = df['ATR'].rolling(window=14).mean()
    df['ATR_vol'] = np.where(df['ATR'] > 1.2 * df['ATR_mavg'], 'High',
                             np.where(df['ATR'] < 0.8 * df['ATR_mavg'], 'Low', 'Medium'))
    
    df['BB_mavg'] = df['BB_width'].rolling(window=20).mean()
    df['BB_vol'] = np.where(df['BB_width'] > 1.2 * df['BB_mavg'], 'High',
                            np.where(df['BB_width'] < 0.8 * df['BB_mavg'], 'Low', 'Medium'))
    
    df['daily_return'] = df['close'].pct_change()
    df['RV'] = df['daily_return'].rolling(window=20).std() * np.sqrt(252)
    rv_80 = df['RV'].quantile(0.8)
    rv_20 = df['RV'].quantile(0.2)
    df['RV_vol'] = np.where(df['RV'] > rv_80, 'High',
                            np.where(df['RV'] < rv_20, 'Low', 'Medium'))
    
    df['Volatility'] = df[['ATR_vol', 'BB_vol', 'RV_vol']].mode(axis=1)[0]
    
    df['Trend'] = np.where(
        (df['MACD'] > df['MACD_signal']) & (df['MACD'] > 0), 'Bullish',
        np.where(
            (df['MACD'] < df['MACD_signal']) & (df['MACD'] < 0), 'Bearish', 'Neutral'
        )
    )
    
    df['Trend_strength'] = np.where(df['ADX'] > 25, 'Strong', 'Weak')
    
    df['Market_Environment'] = df.apply(
        lambda row: f"{row['Volatility']} Vol/{row['Trend']}" if row['Trend_strength'] == 'Strong' else f"{row['Volatility']} Vol/Neutral",
        axis=1
    )

    # Encode categorical data to numeric
    label_encoders = {}
    for column in ['ATR_vol', 'BB_vol', 'RV_vol', 'Volatility', 'Trend', 'Trend_strength', 'Market_Environment']:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        label_encoders[column] = le 

    return df, label_encoders 
def calculate_leg_data(df):
    
    df['percent_delta'] = df['close'].pct_change()
    df = df.reset_index(drop=True)
    
    previous_leg_change = 0
    previous_leg_length = 0
    current_leg_change = 0
    current_leg_length = 0
    
    previous_changes = []
    previous_lengths = []
    current_changes = []
    current_lengths = []
    
    for i in range(len(df)):
        if i == 0:
            current_leg_change = 0
            current_leg_length = 0
        else:
            percent_delta = df.at[i, 'percent_delta']
            if current_leg_length == 0:
                current_leg_change = percent_delta
                current_leg_length = 1
            else:
                if (current_leg_change > 0 and percent_delta > 0) or (current_leg_change < 0 and percent_delta < 0):
                    current_leg_change += percent_delta
                    current_leg_length += 1
                else:
                    previous_leg_change = current_leg_change
                    previous_leg_length = current_leg_length
                    current_leg_change = percent_delta
                    current_leg_length = 1
        previous_changes.append(previous_leg_change)
        previous_lengths.append(previous_leg_length)
        current_changes.append(current_leg_change)
        current_lengths.append(current_leg_length)
    
    df['previous_leg_change'] = previous_changes
    df['previous_leg_length'] = previous_lengths
    df['current_leg_change'] = current_changes
    df['current_leg_length'] = current_lengths
    
    df.drop(columns=['percent_delta'], inplace=True)
    return df

def classify_percent_change(df):
   
    df['percent_change'] = df['close'].pct_change()
    df.dropna(inplace=True)
    
    percentiles = df['percent_change'].quantile([0.05, 0.20, 0.40, 0.60, 0.80, 0.95]).to_dict()
    
    def classify(x, p):
        if x < p[0.05]:
            return 'Down a Lot'
        elif x < p[0.20]:
            return 'Down Moderate'
        elif x < p[0.40]:
            return 'Down a Little'
        elif x < p[0.60]:
            return 'No Change'
        elif x < p[0.80]:
            return 'Up a Little'
        elif x < p[0.95]:
            return 'Up Moderate'
        else:
            return 'Up a Lot'
    
    df['percent_change_classification'] = df['percent_change'].apply(lambda x: classify(x, percentiles))
    return df

def encode_and_scale(df, label_encoders=None, scalers=None):
    
    if label_encoders is None:
        label_encoders = {}
    if scalers is None:
        scalers = {}
    

    categorical_features = ['percent_change_classification', 'Market_Environment']
    for col in categorical_features:
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le  # Save the encoder for future use
  
    numerical_features = [
        'RSI', 'MACD', 'MACD_signal', 'ATR', 'BB_width',
        'previous_leg_change', 'previous_leg_length',
        'current_leg_change', 'current_leg_length'
    ]
    
    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    scalers['numerical'] = scaler  # Save the scaler for future use
    
    return df, label_encoders, scalers

def prepare_sequences(df, seq_length=60, target_steps=1):
   
    FEATURES = [
        'RSI', 'MACD', 'MACD_signal', 'ATR', 'BB_width',
        'previous_leg_change', 'previous_leg_length',
        'current_leg_change', 'current_leg_length',
        'Market_Environment_encoded'
    ]
    
    X = []
    Y_percent_change = []
    Y_leg_direction = []
    
    for i in range(len(df) - seq_length - target_steps + 1):
        seq = df.iloc[i:i + seq_length][FEATURES].values
        target_percent_change = df.iloc[i + seq_length]['percent_change_classification_encoded']
        target_leg_direction = 1 if df.iloc[i + seq_length]['current_leg_change'] > 0 else 0
        
        X.append(seq)
        Y_percent_change.append(target_percent_change)
        Y_leg_direction.append(target_leg_direction)
    
    X = np.array(X)
    Y = {
        'percent_change_classification': np.array(Y_percent_change),
        'leg_direction': np.array(Y_leg_direction, dtype=np.float32).reshape(-1, 1)  # Reshape to (batch_size, 1)
    }
    
    return X, Y

def positional_encoding(seq_length, d_model):
    
    position = np.arange(seq_length)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe = np.zeros((seq_length, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe

def build_enhanced_transformer_model(
    input_shape,
    num_classes_classification,
    head_size=64,
    num_heads=4,
    ff_dim=128,
    num_transformer_blocks=4,
    dropout=0.1,
    l2_reg=1e-4
):
    seq_length, num_features = input_shape
    inputs = Input(shape=input_shape, name='input')
    
    # Feature Embedding
    x = Dense(head_size * num_heads, activation='relu',
              kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(inputs)
    
    # Add Positional Encoding
    pe = positional_encoding(seq_length, head_size * num_heads)
    x += tf.cast(pe, dtype=tf.float32)
    
    # Transformer Blocks
    for i in range(num_transformer_blocks):
        # Layer Normalization
        x_norm = LayerNormalization(epsilon=1e-6, name=f'layer_norm_{i}')(x)
        
        # Multi-Head Self-Attention
        attention = MultiHeadAttention(
            key_dim=head_size,
            num_heads=num_heads,
            dropout=dropout,
            name=f'mha_{i}'
        )(x_norm, x_norm)
        attention = Dropout(dropout)(attention)
        x = Add(name=f'attention_residual_{i}')([x, attention])  # Residual connection
        
        # Layer Normalization
        x_norm = LayerNormalization(epsilon=1e-6, name=f'ffn_layer_norm_{i}')(x)
        
        # Feed-Forward Network
        ffn = Dense(ff_dim, activation='gelu',
                    kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x_norm)
        ffn = Dropout(dropout)(ffn)
        ffn = Dense(head_size * num_heads,
                    kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(ffn)
        ffn = Dropout(dropout)(ffn)
        x = Add(name=f'ffn_residual_{i}')([x, ffn])  # Residual connection

    # Global Pooling and Dense Layers
    x = LayerNormalization(epsilon=1e-6)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout)(x)
    x = Dense(ff_dim, activation='gelu',
              kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
    x = Dropout(dropout)(x)
    x = Dense(64, activation='gelu',
              kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
    x = Dropout(dropout)(x)
    
    # Outputs
    percent_change_output = Dense(
        num_classes_classification,
        activation='softmax',
        name='percent_change_classification'
    )(x)
    leg_direction_output = Dense(
        1,
        activation='sigmoid',
        name='leg_direction'
    )(x)
    
    # Define Model with Named Outputs
    model = Model(
        inputs=inputs, 
        outputs={
            'percent_change_classification': percent_change_output,
            'leg_direction': leg_direction_output
        }
    )
    
    return model

def compile_enhanced_multi_output_model(model, learning_rate=1e-3):
   
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss={
            'percent_change_classification': 'sparse_categorical_crossentropy',
            'leg_direction': 'binary_crossentropy'
        },
        metrics={
            'percent_change_classification': ['accuracy'],
            'leg_direction': ['accuracy']
        }
    )
    return model

def main():
    # Configuration
    BINANCE_API_KEY = 'YOUR_API_KEY'          
    BINANCE_API_SECRET = 'YOUR_API_SECRET'   
    
    SYMBOL = 'BTCUSDT'     
    INTERVAL = '1m'        
    SEQ_LENGTH = 60         
    TARGET_STEPS = 1        
    LIMIT = 5000           
    

    client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
    

    print("Loading data...")
    df = load_data(client, SYMBOL, interval=INTERVAL, limit=LIMIT)
    if df.empty:
        print("No data fetched. Exiting.")
        return
    print("Adding technical indicators...")
    df = add_technical_indicators(df)
    print("Classifying market environments...")
    df, market_env_encoders = classify_market_environments(df)
    print("Calculating leg data...")
    df = calculate_leg_data(df)
    print("Classifying percent changes...")
    df = classify_percent_change(df)
    

    print("Encoding and scaling data...")
    df, label_encoders, scalers = encode_and_scale(df)
    

    print("Preparing sequences...")
    X, Y = prepare_sequences(df, SEQ_LENGTH, TARGET_STEPS)
    

    print(f"Shape of X: {X.shape}")
    for key in Y:
        print(f"Shape of Y[{key}]: {Y[key].shape}")
    

    if len(X) == 0:
        print("Insufficient data after preparing sequences.")
        return
    

    Y_percent_change = Y['percent_change_classification']
    Y_leg_direction = Y['leg_direction']

    assert X.shape[0] == Y_percent_change.shape[0] == Y_leg_direction.shape[0], "Mismatch in sample sizes between X and Y."
    

    print("Verifying label ranges...")
    print(f"percent_change_classification labels: min={Y_percent_change.min()}, max={Y_percent_change.max()}")
    print(f"leg_direction labels: min={Y_leg_direction.min()}, max={Y_leg_direction.max()}")
    
    assert Y_percent_change.min() >= 0 and Y_percent_change.max() < 7, "percent_change_classification labels out of range [0,6]"
    assert Y_leg_direction.min() >= 0 and Y_leg_direction.max() <= 1, "leg_direction labels should be binary (0 or 1)"
    

    from collections import Counter
    counter = Counter(Y_percent_change)
    print("Class distribution for 'percent_change_classification':", counter)
    

    total = sum(counter.values())
    class_weights = {cls: total / (len(counter) * count) for cls, count in counter.items()}
    print("Class Weights:", class_weights)
    

    print("Splitting data into training and testing sets...")
    X_train, X_test, y_percent_change_train, y_percent_change_test, y_leg_direction_train, y_leg_direction_test = train_test_split(
        X,
        Y_percent_change,
        Y_leg_direction,
        test_size=0.2,
        random_state=42,
        stratify=Y_percent_change 
    )
    

    Y_train = {
        'percent_change_classification': y_percent_change_train,
        'leg_direction': y_leg_direction_train
    }
    
    Y_test = {
        'percent_change_classification': y_percent_change_test,
        'leg_direction': y_leg_direction_test
    }
    

    print("Unique labels for 'percent_change_classification':", np.unique(Y_train['percent_change_classification']))
    print("Sample labels:", Y_train['percent_change_classification'][:10])
    print("Unique labels for 'leg_direction':", np.unique(Y_train['leg_direction']))
    print("Sample labels:", Y_train['leg_direction'][:10])
    

    print("Building the Enhanced Transformer model...")
    input_shape = (SEQ_LENGTH, X.shape[2])  # (60, num_features)
    num_classes = len(np.unique(Y_percent_change))  # Should be 7
    model = build_enhanced_transformer_model(
        input_shape=input_shape,
        num_classes_classification=num_classes,
        head_size=64,
        num_heads=4,
        ff_dim=128,
        num_transformer_blocks=4,
        dropout=0.1,
        l2_reg=1e-4
    )
    model = compile_enhanced_multi_output_model(model, learning_rate=1e-3)
   
    

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_percent_change_classification_accuracy',
        patience=10,
        restore_best_weights=True
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_percent_change_classification_accuracy',
        factor=0.5,
        patience=5,
        verbose=1
    )
    

    print("Training the model...")
    history = model.fit(
        X_train,
        {
            'percent_change_classification': Y_train['percent_change_classification'],
            'leg_direction': Y_train['leg_direction']
        },
        epochs=100,
        batch_size=32,
        validation_split=0.2
       
    )
    

    print("Saving label encoders and scalers...")
    with open('label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)
    
    with open('scalers.pkl', 'wb') as f:
        pickle.dump(scalers, f)
    

    print("Saving the trained model...")
    model.save('enhanced_multi_output_transformer_model.h5')
    

    print("Evaluating the model on test data...")
    eval_results = model.evaluate(
        X_test,
        Y_test,
        verbose=0
    )
    for name, value in zip(model.metrics_names, eval_results):
        print(f"{name}: {value:.4f}")
    

    from sklearn.metrics import classification_report, confusion_matrix
    

    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions['percent_change_classification'], axis=1)
    
    print("Classification Report for 'percent_change_classification':")
    print(classification_report(Y_test['percent_change_classification'], predicted_classes))
    

    cm = confusion_matrix(Y_test['percent_change_classification'], predicted_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix for Percent Change Classification')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig("confusion_matrix.png")
    plt.close()
    

    print("Plotting training history...")
    plt.figure(figsize=(12, 8))
    
    # Plot Loss
    plt.subplot(2, 2, 1)
    plt.plot(history.history['percent_change_classification_loss'], label='Train Loss')
    plt.plot(history.history['val_percent_change_classification_loss'], label='Val Loss')
    plt.title('Percent Change Classification Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot Accuracy
    plt.subplot(2, 2, 2)
    plt.plot(history.history['percent_change_classification_accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_percent_change_classification_accuracy'], label='Val Accuracy')
    plt.title('Percent Change Classification Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Loss for Leg Direction
    plt.subplot(2, 2, 3)
    plt.plot(history.history['leg_direction_loss'], label='Train Loss')
    plt.plot(history.history['val_leg_direction_loss'], label='Val Loss')
    plt.title('Leg Direction Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy for Leg Direction
    plt.subplot(2, 2, 4)
    plt.plot(history.history['leg_direction_accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_leg_direction_accuracy'], label='Val Accuracy')
    plt.title('Leg Direction Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("training_history_enhanced.png")
    plt.close()
    
    print("Training complete. Plots saved.")

    return model, history

if __name__ == "__main__":
    model, history = main()
