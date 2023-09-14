def initialize(...):
    # optuna_pickle
    # optuna_figures


def train(model, train_dataloader, val_dataloader, test_dataloader, mean, std, optimizer, optimizer_name_next,
          dl_class, train_dl_args_dict,

          max_epochs, perform_augmix, mixture_width, mixture_depth, augmix_strength, optimizer_name,
          grad_max_norm, scheduler_name, scheduler, loss_function_name, sigmoid_act, softmax_act, to_onehot,
          patience, patience_lr, patience_opt, loss_function, auc_metric, mse_metric, data_aug_p, data_aug_strength,
          max_batch_size, manual_lr,
          momentum, weight_decay, hessian_power, num_batches_per_epoch, use_lookahead, lookahead_k, lookahead_alpha,
          device,

          max_epochs=max_epochs, perform_augmix=perform_augmix, mixture_width=mixture_width,
          mixture_depth=mixture_depth, augmix_strength=augmix_strength, optimizer_name=optimizer_name,
          grad_max_norm=grad_max_norm, scheduler_name=scheduler_name, scheduler=scheduler,
          loss_function_name=loss_function_name, sigmoid_act=sigmoid_act, softmax_act=softmax_act,
          to_onehot=to_onehot, patience=patience, patience_lr=patience_lr, patience_opt=patience_opt,
          loss_function=loss_function, auc_metric=auc_metric, mse_metric=mse_metric,
          data_aug_p=data_aug_p, data_aug_strength=data_aug_strength, max_batch_size=max_batch_size,
          manual_lr=manual_lr, momentum=momentum, weight_decay=weight_decay, hessian_power=hessian_power,
          num_batches_per_epoch=num_batches_per_epoch, use_lookahead=use_lookahead,
          lookahead_k=lookahead_k, lookahead_alpha=lookahead_alpha, device=device,

          logger):


def validate(model, dataloader, mean, std, mode,

             loss_function, loss_function_name, sigmoid_act, softmax_act, auc_metric, mse_metric,
             to_onehot, num_classes, model_name, exp_dir,
             device,

             loss_function=loss_function, loss_function_name=loss_function_name,
             sigmoid_act=sigmoid_act, softmax_act=softmax_act, auc_metric=auc_metric, mse_metric=mse_metric,
             to_onehot=to_onehot, num_classes=num_classes, model_name=model_name, exp_dir=exp_dir,
             device=device,

             logger, save_outputs=True):





