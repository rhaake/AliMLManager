// ##################
// ExtractTrees.C
//
// Example macro to extract trees from several analysis files
// Optional: jet shape branches added
// * Extracts from TList objects in files (usual output from analysis task)
// * Allows extraction into separate trees
// * Faster than merging the files with hadd
// * Computation of meta quantities in ROOT/C++ much faster than in Python

Int_t kMaxNumSamples = 2000000; // stopping condition (no more trees added if threshold reached)


//###############################################################################################
void ExtractTrees()
{
  std::cout << "#### This macro will export samples trees from analysis output ####\n";
  // Example for jet extractor
  if(kTRUE)
  {
    // ############### DATA
    ExtractTreesFromList("13b.root", "AliAnalysisTaskJetExtractorHF_Jet_AKTChargedR040_tracks_pT0150_pt_scheme_RhoR020KT_allJets_histos", "ExtractedJets", "allJets", "./LHC13b.root", "jetshapes");
    ExtractTreesFromList("13c.root", "AliAnalysisTaskJetExtractorHF_Jet_AKTChargedR040_tracks_pT0150_pt_scheme_RhoR020KT_allJets_histos", "ExtractedJets", "allJets", "./LHC13c.root", "jetshapes");
  }
  std::cout << "#### DONE ####\n";
}

//###############################################################################################
void ExtractTreesFromList(const char* fileList, const char* listName, const char* inTreeName, const char* outTreeName, const char* outputFile, const char* newBranches = 0)
{

  TString fileListStr = fileList;
  TObjArray* listArr = fileListStr.Tokenize(" ");
  TList* treeList = new TList;

  std::cout << "Loading trees given in " << fileList << std::endl;
  Int_t summedSamples = 0;
  for(Int_t i=0; i<listArr->GetEntries(); i++)
  {
    // Loading & checking data
    TString fileStr = (static_cast<TObjString*>(listArr->At(i)))->GetString();
    TFile* inputFile = new TFile(fileStr.Data(), "READ");
    if (!inputFile)
    {
      std::cout << "Input file " << fileStr.Data() << " not found!" << std::endl;
      continue;
    }
    TList* tmpList = static_cast<TList*>(inputFile->FindObjectAny(listName));
    if (!tmpList)
    {
      std::cout << "Could not find list " << listName << " in file!" << std::endl;
      continue;
    }
    TTree* inTree = dynamic_cast<TTree*>(tmpList->FindObject(inTreeName));
    if (!inTree)
    {
      std::cout << "Could not find tree " << inTreeName << " in list!" << std::endl;
      continue;
    }

    // Add tree to list if max sample threshold is not reached
    inTree->SetName(outTreeName);
    treeList->Add(inTree);
    summedSamples += inTree->GetEntries();
    delete inputFile;
    std::cout << 100*(i+1)/Float_t(listArr->GetEntries()) << "% of trees loaded, " << summedSamples << " events added." << std::endl;
    if(summedSamples >= kMaxNumSamples)
      break;
  }
  std::cout << "Merging trees... " << std::endl;
  // Write merged tree to output file
  TFile* outFile = new TFile(outputFile,"UPDATE");
  TTree* outTree = TTree::MergeTrees(treeList);
  outTree->SetName(outTreeName);
  outTree->Write(0, TObject::kOverwrite);
  std::cout << "Merging trees... DONE" << std::endl;

  if(newBranches)
  {
    std::cout << "Adding new data branches..." << std::endl;

    // Add new branches on demand
    TString newBranchesStr(newBranches);

    // Selected jet shapes
    if (newBranchesStr.Contains("jetshapes"))
    {
      AddJetShapesToTree(outTree);
      outTree->AutoSave();
    }

    std::cout << "Adding new data branches..." << std::endl;
  }

  delete outFile;
  delete treeList;
}

//###############################################################################################
void AddJetShapesToTree(TTree* tree)
{
  // #### Reserve memory for the values we add to the output tree
  Double_t jetShapeMass = 0;
  Double_t jetShapeDispersion = 0;
  Double_t jetShapeRadial = 0;
  Double_t jetShapeLeSub = 0;
  Double_t jetShapeConstPtDeviation = 0;
  AliBasicJet* inputJet = 0;

  tree->GetBranch("Jets")->SetAddress(&inputJet);

  // #### Add all branches we need for the new tree
  TBranch *branchJetShapeMass = tree->Branch("fJetShapeMass",&jetShapeMass);
  TBranch *branchJetShapeDispersion = tree->Branch("fJetShapeDispersion",&jetShapeDispersion);
  TBranch *branchJetShapeRadial = tree->Branch("fJetShapeRadial",&jetShapeRadial);
  TBranch *branchJetShapeLeSub = tree->Branch("fJetShapeLeSub",&jetShapeLeSub);
  TBranch *branchJetShapeConstPtDeviation = tree->Branch("fJetShapeConstPtDeviation",&jetShapeConstPtDeviation);

  // #### Loop over jets in tree and add important data
  for (Int_t iJet = 0; iJet < tree->GetEntries(); iJet++)
  {
    // Load input into memory
    tree->GetEntry(iJet);
    // Do calculations with inputJet
    GetJetShapes(inputJet, jetShapeMass, jetShapeDispersion, jetShapeLeSub, jetShapeRadial, jetShapeConstPtDeviation);
    // Fill the entries into the tree
    branchJetShapeMass->Fill();
    branchJetShapeDispersion->Fill();
    branchJetShapeRadial->Fill();
    branchJetShapeLeSub->Fill();
    branchJetShapeConstPtDeviation->Fill();

    ShowProgressBar(iJet+1, tree->GetEntries());
  }
  //tree->AutoSave();
}


//###############################################################################################
void GetJetShapes(AliBasicJet* jet, Double_t& jetShapeMass, Double_t& jetShapeDispersion, Double_t& jetShapeLeSub, Double_t& jetShapeRadial, Double_t& jetShapeConstPtDeviation)
{
  Double_t jetCorrectedPt = jet->Pt() - jet->Area() * jet->BackgroundDensity();
  Double_t jetLeadingHadronPt = -999.;
  Double_t jetSubleadingHadronPt = -999.;
  Double_t jetMassP  = 0;
  Double_t jetMassPz = 0;
  Double_t jetDispersionSquareSum = 0;
  Double_t jetDispersionSum = 0;
  

  Double_t jetSquareSum = 0;

  jetShapeRadial = 0;

  for (Int_t i=0;i<jet->GetNumbersOfConstituents();i++)
  {
    AliBasicJetConstituent* constituent = jet->GetJetConstituent(i);
    if(constituent->Pt() > jetLeadingHadronPt)
    {
      jetSubleadingHadronPt = jetLeadingHadronPt;
      jetLeadingHadronPt = constituent->Pt();
    }
    else if(constituent->Pt() > jetSubleadingHadronPt)
      jetSubleadingHadronPt = constituent->Pt();

    Double_t deltaPhi = TMath::Min(TMath::Abs(jet->Phi()-constituent->Phi()),TMath::TwoPi() - TMath::Abs(jet->Phi()-constituent->Phi()));
    Double_t deltaR = TMath::Sqrt( (jet->Eta() - constituent->Eta())*(jet->Eta() - constituent->Eta()) + deltaPhi*deltaPhi );

    // Jet shape calculations
    jetMassP  += constituent->Pt() * TMath::CosH(constituent->Eta());
    jetMassPz += constituent->Pt() * TMath::SinH(constituent->Eta());
    jetDispersionSum += constituent->Pt();
    jetSquareSum += constituent->Pt()*constituent->Pt();

    if(jetCorrectedPt)
      jetShapeRadial += constituent->Pt()/jetCorrectedPt * deltaR;
      
  }

  if(jetMassP*jetMassP - jetCorrectedPt*jetCorrectedPt - jetMassPz*jetMassPz > 1e-50)
    jetShapeMass = TMath::Sqrt(jetMassP*jetMassP - jetCorrectedPt*jetCorrectedPt - jetMassPz*jetMassPz);
  if(jet->GetNumbersOfConstituents() > 1)
    jetShapeLeSub = jetLeadingHadronPt - jetSubleadingHadronPt;
  else
    jetShapeLeSub = 1.;

  if(jetDispersionSum)
    jetShapeDispersion = TMath::Sqrt(jetSquareSum)/jetDispersionSum;
  else
    jetShapeDispersion = 0;

  if(jet->GetNumbersOfConstituents() && jetCorrectedPt)
    jetShapeConstPtDeviation = jetSquareSum / jet->GetNumbersOfConstituents() / jetCorrectedPt;
  else
    jetShapeConstPtDeviation = 0;
}

//###############################################################################################
void ShowProgressBar(Double_t current, Double_t maximum)
{
  static Int_t lastPercentageShown = -1;
  Int_t currentPercentage = static_cast<Int_t>(100.*(current/maximum));
  if(currentPercentage != lastPercentageShown)
  {
    std::cout << Form("%i %% done.", currentPercentage)  << std::endl;
    lastPercentageShown = currentPercentage;
  }
}


//###############################################################################################
TString GetStringFromFilePattern(const char* inPattern, Int_t startNum, Int_t endNum)
{
  TString list("");
  for(Int_t i=startNum; i<=endNum; i++)
  {
    list += Form(inPattern, i);
    list += " ";
  }
  return list;
}
