// ##################
// AnalysisBTagging.C
//
// b-jet tagging analysis macro


//###############################################################################################
void AnalysisBTagging()
{
  std::cout << "#### This macro will export samples trees from analysis output ####\n";

  // Load data from tree
  TFile* inFile = new TFile("../Results/ScoredPt.root", "READ");
  TTree* inTree_bjets = static_cast<TTree*>(inFile->Get("Testing_bJets"));
  TTree* inTree_cjets = static_cast<TTree*>(inFile->Get("Testing_cJets"));
  TTree* inTree_lightjets = static_cast<TTree*>(inFile->Get("Testing_lightJets"));

  // Calculate thresholds to obtain given efficiencies
  std::cout << "Calculating correct score thresholds...\n";
  Double_t* thresholds = GetScoreThresholds(inTree_bjets);

  ////////////////////////////////////////////////////////////
  // Plot mistagging efficiencies with these thresholds
  TH1* hTaggingUDSG = GetHistogramTaggingEfficiency("hTaggingUDSG", inTree_lightjets, thresholds);
  TH1* hTaggingC    = GetHistogramTaggingEfficiency("hTaggingC", inTree_cjets, thresholds);
  TH1* hTaggingB    = GetHistogramTaggingEfficiency("hTaggingB", inTree_bjets, thresholds);

  TH1* hTaggingC_ref    = GetHistogramTaggingEfficiencyRef(kTRUE);
  TH1* hTaggingUDSG_ref = GetHistogramTaggingEfficiencyRef(kFALSE);

  hTaggingUDSG->SetMarkerColor(kBlue);
  hTaggingUDSG->SetLineColor(kBlue);
  hTaggingC->SetMarkerColor(kBlack);
  hTaggingC->SetLineColor(kBlack);

  hTaggingUDSG_ref->SetMarkerColor(kBlue);
  hTaggingUDSG_ref->SetLineColor(kBlue);
  hTaggingC_ref->SetMarkerColor(kBlack);
  hTaggingC_ref->SetLineColor(kBlack);

  hTaggingUDSG_ref->SetLineStyle(0);
  hTaggingC_ref->SetLineStyle(0);
  hTaggingUDSG_ref->SetMarkerStyle(23);
  hTaggingC_ref->SetMarkerStyle(23);

  hTaggingB->SetMarkerColor(kRed);
  hTaggingB->SetLineColor(kRed);

  TCanvas* c = new TCanvas();
  c->SetLogy();
  hTaggingUDSG->GetYaxis()->SetRangeUser(2e-4, 40);
  hTaggingUDSG->Draw("e1p");

  hTaggingC->Draw("e1p same");
  hTaggingB->Draw("e1p same");

  hTaggingUDSG_ref->Draw("e1p same");
  hTaggingC_ref->Draw("e1p same");

  c->SaveAs("./MistaggingEfficiencies.png");

  std::cout << "#### DONE ####\n";
}


//###############################################################################################
TH1* GetHistogramTaggingEfficiencyRef(Bool_t isCharm)
{
  Double_t ptBinning[7]    = {20, 25, 30, 35, 40, 45, 50};
  Double_t refLightJets[6] = {0.000573, 0.000751, 0.000873, 0.001078, 0.00111, 0.001253};
  Double_t refCJets[6]     = {0.0287, 0.0376, 0.0493, 0.0493, 0.05395, 0.0609};

  const char* name = isCharm ? "refTaggingEffCharm" : "refTaggingEffLight";
  TH1* hTagging = new TH1D(name, name, 6, ptBinning);
  hTagging->Sumw2();

  for(Int_t i=1; i<hTagging->GetNbinsX();i++)
  {
    hTagging->SetBinContent(i, isCharm ? refCJets[i-1] : refLightJets[i-1]);
    hTagging->SetBinError(i, 0);
  }

  return hTagging;
}


//###############################################################################################
TH1* GetHistogramTaggingEfficiency(const char* name, TTree* tree, Double_t* thresholds)
{
  Double_t ptBinning[13]    = {15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100};
  Double_t score = 0;
  Double_t pt = 0;

  // Plot mistagging efficiencies with these thresholds
  TH1* hTagging = new TH1D(name, name, 12, ptBinning);
  hTagging->Sumw2();
  TH1* hSamples = new TH1D(Form("%s_samples",name), name, 12, ptBinning);
  hSamples->Sumw2();

  tree->SetBranchAddress("Score", &score);
  tree->SetBranchAddress("Pt", &pt);
  Int_t nTot = 0;
  for(Int_t i=0; i<tree->GetEntries(); i++)
  {
    tree->GetEntry(i);
    Int_t ptBin = hTagging->GetXaxis()->FindBin(pt);

    if(score < thresholds[ptBin-1])
      hTagging->Fill(pt);

    hSamples->Fill(pt);
  }

  hTagging->Divide(hSamples);

  return hTagging;
}


//###############################################################################################
Double_t* GetScoreThresholds(TTree* tree)
{
  // --> Sort the tree wrt score and fill the selected efficiency
  Double_t score = 0;
  Double_t pt = 0;
  Long64_t nEntries = tree->GetEntries();

  tree->SetBranchAddress("Score", &score);
  tree->SetBranchAddress("Pt", &pt);
  tree->Draw("Score","","goff");
  Int_t *index = new Int_t[nEntries];
  TMath::Sort(nEntries, tree->GetV1(), index);

  Double_t* thresholds       = new Double_t[13];
  for(Int_t i=0;i<13;i++) thresholds[i] = 0;
  Int_t     ptBinning[13]    = {15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100};
  Double_t  efficiencies[13] = {0.1745, 0.1745, 0.1968, 0.21, 0.215, 0.23, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24};
  Long64_t  accepted[13]     = {0,0,0,0,0,0,0,0,0,0,0,0,0};
  Long64_t  totalAccepted[13];
  for(Int_t i=0; i<12; i++)
    totalAccepted[i] = tree->GetEntries(Form("Pt>=%d && Pt<%d", ptBinning[i], ptBinning[i+1]));

  Int_t done = 0;
  for(Int_t i=nEntries-1; i>0; i--)
  {
    tree->GetEntry(index[i]);

    for(Int_t ptbin=0; ptbin<12; ptbin++)
    {
      if(thresholds[ptbin])
        continue;
      if((pt >= ptBinning[ptbin]) && (pt < ptBinning[ptbin+1]))
        accepted[ptbin]++;

      if(accepted[ptbin] >= efficiencies[ptbin]*totalAccepted[ptbin])
      {
        std::cout << Form("Threshold for Pt=%2d-%2d (eff=%2.2f), threshold=%f", ptBinning[ptbin], ptBinning[ptbin+1], efficiencies[ptbin], score) << std::endl;
        thresholds[ptbin] = score;
        done++;
      }
    }
    if(done >= 12)
      break;
  }

  return thresholds;

}
