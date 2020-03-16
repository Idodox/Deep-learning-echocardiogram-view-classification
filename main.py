from comet_ml import Experimentimport torchimport torch.optim as optimimport torch.nn as nnfrom torchvision import transformsfrom functools import partialfrom tqdm import tqdmimport osfrom modular_cnn import ModularCNN, make_layersfrom torchsummary import summaryfrom torchutils import pickle_loader, DatasetFolderWithPaths, Normalize, \    ToTensor, train, evalfrom utils import get_train_val_idxtorch.backends.cudnn.benchmark=Trueprint('CUDA available:', torch.cuda.is_available())print('CUDA enabled:', torch.backends.cudnn.enabled)# torch.cuda.empty_cache()log_data = Falseexperiment = NoneHP1 = {"learning_rate": 0.00001               ,"n_epochs": 80               ,"batch_size": 64               ,"num_workers": 6               ,"normalized_data": True               ,"stratified": False               ,"horizontal_flip": False               ,"max_frames": 10               ,"random_seed": 999               ,"flip_prob": 0.5               ,"dataset": "cont3frame_steps"               ,"resolution": 100               ,"adaptive_pool": (7, 6, 6)               ,"features": [16,16,"M",16,16,"M",32,32,"M"]               ,"classifier": [0.4, 50, 0.2, 25]                }for hyper_params in [HP1]:    model = ModularCNN(make_layers(hyper_params["features"], batch_norm=True), classifier = hyper_params["classifier"], adaptive_pool=hyper_params["adaptive_pool"])    if torch.cuda.is_available():        model = model.cuda()    # Log number of parameters    hyper_params['trainable_params'] = sum(p.numel() for p in model.parameters())    print('N_trainable_params:', hyper_params['trainable_params'])    data_transforms = transforms.Compose([        ToTensor()        ,Normalize(0.213303, 0.21379)        # ,RandomHorizontalFlip(hyper_params["flip_prob"])    ])    # ROOT_PATH = str("/home/ido/data/" + hyper_params['dataset'])    ROOT_PATH = str('/Users/idofarhi/Documents/Thesis/Data/frames/' + hyper_params['dataset'])    master_data_set = DatasetFolderWithPaths(ROOT_PATH                                    , transform = data_transforms                                    , loader = partial(pickle_loader, min_frames = hyper_params['max_frames'])                                    , extensions = '.pickle'                                    )    train_idx, val_idx = get_train_val_idx(master_data_set, random_state = hyper_params['random_seed'], test_size = 0.2)    train_set = torch.utils.data.Subset(master_data_set, train_idx)    val_set = torch.utils.data.Subset(master_data_set, val_idx)    train_loader = torch.utils.data.DataLoader(train_set                                         , batch_size=hyper_params['batch_size']                                         , shuffle=True                                         # ,batch_sampler =  # TODO: add stratified sampling                                         , num_workers=hyper_params['num_workers']                                         , drop_last=False                                         )    # online_mean_and_std(train_loader)    val_loader = torch.utils.data.DataLoader(val_set                                         , batch_size=hyper_params['batch_size']                                         , shuffle=True                                         # ,batch_sampler =  # TODO: add stratified sampling                                         , num_workers=hyper_params['num_workers']                                         , drop_last=False                                         )    optimizer = optim.Adam(model.parameters(), lr=hyper_params['learning_rate'])    criterion = nn.CrossEntropyLoss()    log_number_train = log_number_val = 0    if log_data:        # Comet ML experiment        experiment = Experiment(api_key="BEnSW6NdCjUCZsWIto0yhxts1" ,project_name="thesis" ,workspace="idodox")        experiment.log_parameters(hyper_params)    summary(model, (1, hyper_params["max_frames"], hyper_params["resolution"], hyper_params["resolution"]))    if torch.cuda.device_count() > 1:      print("Let's use", torch.cuda.device_count(), "GPUs!")      # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs      model = nn.DataParallel(model)    for epoch in tqdm(range(hyper_params["n_epochs"])):        train(epoch, train_loader, optimizer, criterion, log_data, experiment, model, log_number_train)        eval(epoch, train_loader, optimizer, criterion, log_data, experiment, model, log_number_test)print("Saving model...")torch.save(model.state_dict(), os.getcwd() + "/model.pt")