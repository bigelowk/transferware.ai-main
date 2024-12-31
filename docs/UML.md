# System UML Diagrams

## Sequence

### Query image
```mermaid
sequenceDiagram
    autonumber
    
    actor user
    
    user->>+tcc-ui: Upload image of sherd
    tcc-ui->>+query-api: Submit image for query
    query-api->>+model: model :query(image)
    model-->+preprocessed-data: Use preprocessed data to match image
    model->>-query-api: IDs of top k matches and confidence
    query-api->>-tcc-ui: IDs of top k matches and confidence
    loop For each image ID 
        tcc-ui->>+tcc-api: Request record for ID
        tcc-api->>-tcc-ui: Image and metadata
    end
    tcc-ui->>-user: Top k records displayed
```

### Update Model
```mermaid
sequenceDiagram
    autonumber
    actor system-trigger
    
    system-trigger->>training-script: Start job (each week etc.)
    training-script->>+tcc-api-cache: :open_cache(directory)
    alt If cache not updated 
        tcc-api-cache->>+tcc-api: Get all images and metadata
        tcc-api->>-tcc-api-cache: 
    end
    tcc-api-cache->>-training-script: Cache wrapper object
    training-script->>+model-trainer: :train(cache)
    model-trainer->>-training-script: Model and caches
    training-script->>+validator: :validate(model, cache, validation_dataset)
    validator->>-training-script: Validation percent
    alt If validation percent high enough 
        training-script->>+query-api: Send model and caches to update endpoint in tar archive
        query-api->>preprocessed-data: Replace with updated data
        
        alt If cache not updated 
        query-api->>tcc-api-cache: :reload_api_cache()
        tcc-api-cache->>+tcc-api: Get all images and metadata
        tcc-api->>-tcc-api-cache: 
        tcc-api-cache->>query-api: 
        end
        
        query-api->>mqtt-broker: Publish /transferware/reload
        mqtt-broker->>query-api: Forward message, reloading local model and cache
    end
    
```

## Communication

### Query image
````mermaid
stateDiagram-v2
    direction LR
    
    state tcc_api
    state query_api
    state model
    state preprocessed-data
    state tcc_ui
    state training_script 
    state tcc_api_cache
    state model-trainer
    state validator
    state user
    
    user-->tcc_ui: 1 Upload image of sherd
    tcc_ui-->query_api: 1.1 Submit image for query
    query_api-->model: 1.1.1 query(image)
    model-->preprocessed_data: 1.1.1.1 retrieve(dir)
    model--> query_api: 1.1.2 IDs of top k matches and confidence
    query_api-->tcc_ui: 1.2 IDs of top k matches and confidence
    
    tcc_ui-->tcc_api: 1.3 Request record for ID
    tcc_api-->tcc_ui: 1.4 Image and metadata
    
    tcc_ui-->user: 1.5 Tok k records displayed
````

### Update model
````mermaid
stateDiagram-v2
    direction LR
    
    state tcc_api
    state query_api
    state model
    state preprocessed_data
    state tcc_ui
    state training_script 
    state tcc_api_cache
    state model_trainer
    state validator
    state system_trigger
    
    system_trigger-->training_script: 1 Start job
    training_script-->tcc_api_cache: 1.1 open_cache(directory)
    tcc_api_cache-->tcc_api: 1.1.1 Get all images and metadata
    tcc_api-->tcc_api_cache: 1.1.2 Images and metadata
    tcc_api_cache-->training_script: 1.2 Cache wrapper object
    training_script-->model_trainer: 1.3 train(cache)
    model_trainer-->training_script: 1.4 Model and caches
    training_script-->validator: 1.5 validator(model, cache, validation_dataset)
    validator-->training_script: 1.6 Validation percent
    training_script-->query_api: 1.7 Send model and caches to update endpoint
    query_api-->preprocessed_data: 1.7.1 Replace with updated data 
    query_api-->model: 1.7.2 reload()
````

# Classes

## Model

```mermaid
classDiagram
    class ImageMatch {
        + id: int
        + confidence: float
    }
    
    class Model {
        <<interface>>
        + query(image_tensor: Tensor) list~ImageMatch~
        + get_resource_files() list~Path~
    }
    
    class Trainer {
        <<interface>>
        + train(dataset: CacheDataset) Model
    }
    
    class Validator {
        <<interface>>
        + validate(model: Model, validation_data: ImageFolder) tuple~dict[int, float], float~
    }
    
    class AbstractModelFactory {
        <<abstract>>
        - resource_dir: Path
        + AbstractModelFactory(resource_dir: Path)
        + get_model() Model
        + get_trainer() Trainer
        + get_validator() Validator
    }
    
    Model *-- ImageMatch
    AbstractModelFactory *-- Model
    AbstractModelFactory *-- Trainer
    AbstractModelFactory *-- Validator
```
An abstract factory is used to create working combinations of pipelines. The Model is either loaded from resources on
disk, or created by the Trainer. The validator takes the model and does black box validation by checking if validation
images (from Wayne state) are in the top 10 query results. So this validation is not for whatever the trainer does, 
but for the system as a whole.

The end application should load a specific factory based off configurations, so ideally all model information will be
encapsulated within these three classes. This should make it trivial to then swap in something like SIFT matching 
instead of a deep learning approach, while still allowing enough complexity for embedding approaches.

## Embeddings model

```mermaid
classDiagram
    class ImageMatch {
        + id: int
        + confidence: float
    }

    class Model {
        <<interface>>
        + query(image_tensor: Tensor) list~ImageMatch~
        + get_resource_files() list~Path~
    }

    class Trainer {
        <<interface>>
        + train(dataset: CacheDataset) Model
    }

    class Validator {
        <<interface>>
        + validate(model: Model, validation_data: ImageFolder) float
    }

    class AbstractModelFactory {
        <<abstract>>
        - resource_dir: Path
        + AbstractModelFactory(resource_dir: Path)
        + get_model() Model
        + get_trainer() Trainer
        + get_validator() Validator
    }

    class ZhaoModelFactory
    class ZhaoModel {
        + make_tensorboard_projection(ata: CacheDataset, sample_size: int)
    }
    class EmbeddingsModelImplementation {
        <<abstract>>
        + transform(self, img: Tensor | Image)*
        + training_mode()
        + eval_mode()
        + add_augmentations(self, augmentations: transforms.Transform)
        + get_embedding(self, image: Tensor | Image)*
    }
    class ZhaoTrainer {
        + generate_annoy_cache(model: EmbeddingsModelImplementation, ds: CacheDataset, visitor: Optional[Callable] = None) tuple~annoy.AnnoyIndex, list[int]~
    }
    class GenericValidator
    class AnnoyIndex

    Model *-- ImageMatch
    AbstractModelFactory *-- Model
    AbstractModelFactory *-- Trainer
    AbstractModelFactory *-- Validator
    AbstractModelFactory <|-- ZhaoModelFactory
    Model <|-- ZhaoModel
    Validator <|-- GenericValidator
    ZhaoModel *-- EmbeddingsModelImplementation
    ZhaoModel *-- AnnoyIndex
    EmbeddingsModelImplementation <|-- ZhaoVGGModel
    EmbeddingsModelImplementation <|-- SwinModel
    EmbeddingsModelImplementation <|-- ConvnextModel
    EmbeddingsModelImplementation <|-- ResNetModel
    Trainer <|--ZhaoTrainer
    ZhaoTrainer *-- EmbeddingsModelImplementation
```
The embeddings approach uses a bridge to decouple the low level torch implementations that create the embeddings
themselves from the high level embeddings search logic. This allows us to easily experiment with different underlying
NN architectures without needing to change the rest of the code. This is not really for runtime behavior, just
structure.
## Dataset
```mermaid
classDiagram
namespace torch {
    class Dataset { 
        <<interface>>
        __getitem__(idx: int)*
        __len__()*
    }
}
namespace DataHandling {
    class ApiCache {
        -directory: Path
        -cache_file: Path
        -assets_dir: Path
        + ApiCache(dir: Path)
        + ensure_cached() 
        + as_df() polars.Dataframe
    }

    class CacheDataset {
        -cache: ApiCache
        + CacheDataset(dir: Path)
        + class_labels() list~str~
        __getitem__(idx: int) tuple~Tensor, Tensor~
        __len__() int
    }
    
}
    Dataset <|-- CacheDataset
    CacheDataset *-- ApiCache

```

The dataset is backed by a wrapper over the API cache, where the API cache is represented as a dataframe.
This setup allows for us to easily make drastic changes in class choice, as we don't need to change a directory layout,
just change some dataframe queries. Get item is returning image, class_id pairs.
