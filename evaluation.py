import torch

# SR : Segmentation Result
# GT : Ground Truth

def get_accuracy(SR,GT,threshold=0.3):
    SR[SR<=threshold]=0
    SR=torch.argmax(SR,1)
    SR=torch.nn.functional.one_hot(SR,7).permute(0,3,1,2)
    SR2=SR[:,1:,:,:]
    GT2= (GT>0)[:,1:,:,:]
    
    TP = ((SR2==True)&(GT2==True)).sum((2,3)).double()
    FP = ((SR2==True)&(GT2==False)).sum((2,3)).double()
    TN = ((SR2==False)&(GT2==False)).sum((2,3)).double()
    FN = ((SR2==False)&(GT2==True)).sum((2,3)).double()
    Union=TP+FP+FN
    
    SE = (TP/(TP+FN+1e-4))
    PC = (TP/(TP+FP + 1e-4))
    
    F1 = (2*SE*PC/(SE+PC + 1e-4)).mean(1).sum().item()
    SE = SE.mean(1).sum().item()
    PC = PC.mean(1).sum().item()
    SP = (TN/(TN+FP + 1e-4)).mean(1).sum().item()
    JS = (TP/(Union + 1e-4)).mean(1).sum().item()
    DC = (2*TP/(Union+TP + 1e-4)).sum(0).cpu().detach().numpy()
    
    acc=(SR2==GT2).sum().item()/(SR2.size(1)*SR2.size(2)*SR.size(3))
    return acc,SE,SP,PC,F1,JS,DC
    
    
def get_sensitivity(SR,GT,threshold=0.5):
    # Sensitivity == Recall
    total_SEN=0
    for ii in range(0,SR.shape[0]):
        SR2 = SR[ii, 1:, :, :].detach().clone()
        GT2 = GT[ii, 1:, :, :].detach().clone()
        SR2 = SR2 > threshold
        GT2 = GT2 == torch.max(GT2)
        TP = ((SR2==1).byte()+(GT2==1).byte())==2
        FN = ((SR2==0).byte()+(GT2==1).byte())==2
        SE = float(torch.sum(TP))/(float(torch.sum(TP)+torch.sum(FN)) + 1e-6)  
        total_SEN+=SE  
    
    return total_SEN

def get_specificity(SR,GT,threshold=0.5):

    total_SPE=0
    for ii in range(0,SR.shape[0]):
        SR2 = SR[ii, 1:, :, :].detach().clone()
        GT2 = GT[ii, 1:, :, :].detach().clone()
        SR2 = SR2 > threshold
        GT2 = GT2 == torch.max(GT2)

        # TN : True Negative
        # FP : False Positive
        TN = ((SR2==0).byte()+(GT2==0).byte())==2
        FP = ((SR2==1).byte()+(GT2==0).byte())==2

        SP = float(torch.sum(TN))/(float(torch.sum(torch.sum(TN)+torch.sum(FP))) + 1e-6)
        total_SPE+=SP 
    
    return total_SPE

def get_precision(SR,GT,threshold=0.5):

    total_PC=0
    for ii in range(0,SR.shape[0]):
        SR2 = SR[ii, 1:, :, :].detach().clone()
        GT2 = GT[ii, 1:, :, :].detach().clone()
        SR2 = SR2 > threshold
        GT2 = GT2 == torch.max(GT2)
        TP = ((SR2==1).byte()+(GT2==1).byte())==2
        FP = ((SR2==1).byte()+(GT2==0).byte())==2

        PC = float(torch.sum(TP))/(float(torch.sum(TP)+torch.sum(FP)) + 1e-6)
        total_PC+=PC

    return total_PC

def get_F1(SR,GT,threshold=0.5):
    # Sensitivity == Recall
    total_F1=0
    for ii in range(0,SR.shape[0]):
        SR2 = SR[ii, 1:, :, :].detach().clone()
        GT2 = GT[ii, 1:, :, :].detach().clone()
        SR2 = SR2 > threshold
        GT2 = GT2 == torch.max(GT2)
        TP = ((SR2==1).byte()+(GT2==1).byte())==2
        FN = ((SR2==0).byte()+(GT2==1).byte())==2
        FP = ((SR2==1).byte()+(GT2==0).byte())==2
        SE = float(torch.sum(TP))/(float(torch.sum(TP)+torch.sum(FN)) + 1e-6) 
        PC = float(torch.sum(TP))/(float(torch.sum(TP)+torch.sum(FP)) + 1e-6) 

        F1 = 2*SE*PC/(SE+PC + 1e-6)
        total_F1+=F1

    return total_F1


def get_JS2(SR,GT,threshold=0.5):
    total_JS=0
    for ii in range(0,SR.shape[0]):
        SR2 = SR[ii, 1:, :, :].detach().clone()
        GT2 = GT[ii, 1:, :, :].detach().clone()
        SR2 = SR2 > threshold
        GT2 = GT2 == torch.max(GT2)
        
        Inter = torch.sum((SR2.byte()+GT2.byte())==2)
        Union = torch.sum((SR2.byte()+GT2.byte())>=1)
        #print(f'JS: {SR.any()}, {GT.any()}')
        #print(f'JS: {Inter}, {Union}')
        
        JS = float(Inter)/(float(Union) + 1e-6)
        total_JS+=JS
    
    return total_JS


def get_DC(SR,GT,threshold=0.5):
    # DC : Dice Coefficient
    # batch > 1
    total_DC=0
    for ii in range(0,SR.shape[0]):
        SR2 = SR[ii, 1:, :, :].detach().clone()
        GT2 = GT[ii, 1:, :, :].detach().clone()
        SR2 = SR2 > threshold
        GT2 = GT2 == torch.max(GT2)

        Inter = torch.sum((SR2.byte()+GT2.byte())==2)
        DC = float(2*Inter)/(float(torch.sum(SR2)+torch.sum(GT2)) + 1e-6)
        total_DC+=DC

    return total_DC

def get_DC2(SR,GT,threshold=0.5):
    # DC : Dice Coefficient
    total_DC=0
    for ii in range(0,SR.shape[0]):
        SR2 = SR[ii, 3, :, :].detach().clone()
        GT2 = GT[ii, 3, :, :].detach().clone()
        SR2 = SR2 > threshold
        GT2 = GT2 == torch.max(GT2)

        Inter = torch.sum((SR2.byte()+GT2.byte())==2)
        DC = float(2*Inter)/(float(torch.sum(SR2)+torch.sum(GT2)) + 1e-6)
        total_DC+=DC

    return total_DC





