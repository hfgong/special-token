def evaluate_mae_reconstruction(model, dataloader, device):
    """Comprehensive evaluation of MAE reconstruction quality."""
    model.eval()
    
    total_mse = 0
    total_psnr = 0
    total_ssim = 0
    num_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            
            # Forward pass
            x_masked, mask, ids_restore = random_masking(
                model.patch_embed(batch)
            )
            pred = model(x_masked, mask, ids_restore)
            
            # Convert predictions back to images
            pred_images = model.unpatchify(pred)
            
            # Compute metrics
            mse = F.mse_loss(pred_images, batch)
            psnr = compute_psnr(pred_images, batch)
            ssim = compute_ssim(pred_images, batch)
            
            total_mse += mse.item()
            total_psnr += psnr.item()
            total_ssim += ssim.item()
            num_samples += 1
    
    return {
        'mse': total_mse / num_samples,
        'psnr': total_psnr / num_samples,
        'ssim': total_ssim / num_samples
    }

def compute_psnr(pred, target):
    """Compute Peak Signal-to-Noise Ratio."""
    mse = F.mse_loss(pred, target)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr

def compute_ssim(pred, target):
    """Compute Structural Similarity Index."""
    # Implementation using kornia or custom SSIM
    from kornia.losses import ssim_loss
    return 1 - ssim_loss(pred, target, window_size=11)