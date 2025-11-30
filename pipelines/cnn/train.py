from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC


def build_and_compile()->Model:
    base_model = MobileNetV2(weights='imagenet', include_top=False,input_shape=(224,224,3))
    base_model.trainable = False
    x=base_model.output
    x=GlobalAveragePooling2D()(x)
    x=Dense(256,activation='relu')(x)
    x=Dropout(0.5)(x)
    output=Dense(1,activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001),loss='binary_crossentropy',metrics=['accuracy',AUC(name='auc')])
    return model


def train(train_data,validation_data,checkpoint_path):
    checkpoints = ModelCheckpoint(checkpoint_path,save_best_only=True,)
    early_stopping = EarlyStopping(patience=10,restore_best_weights=True)
    rd=ReduceLROnPlateau(patience=10)
    model =build_and_compile()
    model.fit(train_data,batch_size=16,epochs=50,validation_data=validation_data,callbacks=[checkpoints,early_stopping,rd])
    return model
    
    

