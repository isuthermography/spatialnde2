#include <unordered_map>
#include <cstring>
#include <vector>
#include <cmath>
#include <regex>

#ifndef SNDE_UNITS_HPP
#define SNDE_UNITS_HPP

namespace snde {


static inline std::string stripstr(std::string inp)
  {

    std::string out;
    
    size_t startpos=inp.find_first_not_of(" \t\r\n");
    size_t endpos=inp.find_last_not_of(" \t\r\n");
    
    if (endpos == std::string::npos) {
      // no non-whitespace characters
      out="";
      return out;
    }

    assert(startpos != std::string::npos);
    out=inp.substr(startpos,endpos-startpos);

    return out; 
  }

  class UnitDef {
  public:
    std::string MeasName; // name of quantity measured by this unit
    std::string SingularName; // singular name of unit (e.g. meter)
    std::string PluralName; // plural name of unit (e.g. meters)
    std::vector<std::string> AbbrevList; // list of abbreviations
    size_t Index; // index of creation of this unit
    double PreferredPower; // Preferred power of 10; ... typically 0, but 3 for grams indicating usual use of kh
    bool SiPrefixFlag;

    UnitDef(std::string MeasName,std::string SingularName,std::string PluralName,std::vector<std::string> AbbrevList,size_t Index,double PreferredPower,bool SiPrefixFlag) :
      MeasName(MeasName),
      SingularName(SingularName),
      PluralName(PluralName),
      AbbrevList(AbbrevList),
      Index(Index),
      PreferredPower(PreferredPower),
      SiPrefixFlag(SiPrefixFlag)
    {

    }
    
  };
  
  
  /* simplified definition found below 
  class SIPrefix {
  public:
    std::string Prefix;
    std::string Abbrev;
    int Power;

    SIPrefix(std::string Prefix,std::string Abbrev,int Power) :
      Prefix(Prefix),
      Abbrev(Abbrev),
      Power(Power)
    {

    }
  };

  static const std::vector<SIPrefix> SIPrefixes = {
						   { "Yotta","Y",24 },
						   { "Zetta", "Z", 21},
						   { "Exa", "E", 18},
						   {"Peta", "P", 15},
						   {"Tera", "T", 12},
						   {"Giga", "G", 9},
						   {"Mega", "M", 6},
						   {"kilo", "k", 3},
						   {"milli", "m", -3},
						   {"micro", "u", -6}, 
						   {"nano", "n", -9}, 
						   {"pico", "p", -12},
						   {"femto", "f", -15}, 
						   {"atto", "a", -18},
						   {"zepto", "z", -21}, 
						   {"yocto", "y", -24}
  };

  */
  
  class UnitFactor {
  public:
    std::shared_ptr<UnitDef> Unit;
    std::string NameOnly;
    double Power; // e.g. 1 for meters, 2 for meters^2, positive for numerator... COEFFICIENT WILL BE TAKEN TO THIS POWER TOO!
    double Coefficient; // Coefficient of these units. When normalized will be 1.0, except if there is a PreferredPower in which case this will be 10^PreferredPower

    UnitFactor(std::shared_ptr<UnitDef> Unit,std::string NameOnly, double Power, double Coefficient) :
      Unit(Unit),
      NameOnly(NameOnly),
      Power(Power),
      Coefficient(Coefficient)
    {

    }

    friend int Compare(const UnitFactor &FactorA, const UnitFactor &FactorB)
    // returns < 0 for A < B, 0 for A==B, > 0  for A > B, computed according to unit index # and
    // alphabetical order for unknown units 
    {
      if (FactorA.Unit && !FactorB.Unit) {
	return -1;
      }

      if (FactorB.Unit && !FactorA.Unit) {
	return 1;
      }

      if (FactorA.Unit && FactorB.Unit) {
	if (FactorA.Unit->Index < FactorB.Unit->Index) {
	  return -1;
	}
	if (FactorA.Unit->Index == FactorB.Unit->Index) {
	  return 0;
	}
	if (FactorA.Unit->Index > FactorB.Unit->Index) {
	  return 1; 
	}
	assert(0); // unreachable
      }
      if (!FactorA.Unit && !FactorB.Unit) {
	return strcmp(FactorA.NameOnly.c_str(),FactorB.NameOnly.c_str());
      }
      assert(0); // unreachable
      
      return 0;
    }
    
    friend bool operator==(const UnitFactor &lhs, const UnitFactor &rhs) {
      return Compare(lhs,rhs) == 0;
    }
    friend bool operator!=(const UnitFactor &lhs, const UnitFactor &rhs) {
      return Compare(lhs,rhs) != 0;
    }
    friend bool operator<(const UnitFactor &lhs, const UnitFactor &rhs) {
      return Compare(lhs,rhs) < 0;
    }
    friend bool operator>(const UnitFactor &lhs, const UnitFactor &rhs) {
      return Compare(lhs,rhs) > 0;
    }
    friend bool operator<=(const UnitFactor &lhs, const UnitFactor &rhs) {
      return Compare(lhs,rhs) <= 0;
    }
    friend bool operator>=(const UnitFactor &lhs, const UnitFactor &rhs) {
      return Compare(lhs,rhs) >= 0;
    }
  };


static inline  std::tuple<std::vector<UnitFactor>,double> sortunits(std::vector<UnitFactor> Factors)
  {
    // Pull out all coefficients out in front
    double Coefficient = 1.0;
    for (auto & Factor: Factors) {
      
      Coefficient *= pow(Factor.Coefficient,Factor.Power);
      Factor.Coefficient=1.0;
    }
    // Perform sort
    std::sort(Factors.begin(),Factors.end());

    // Combine like units!
    size_t pos=0;

    while (Factors.size() > 0 && pos < Factors.size()-1) {
      if (Factors[pos]==Factors[pos+1]) {
	Factors[pos].Power += Factors[pos+1].Power;
	Factors.erase(Factors.begin()+pos+1);
	
	if (fabs(Factors[pos].Power) < 1e-8) {
	  // units canceled!
	  Factors.erase(Factors.begin()+pos);
	}
      } else {
	pos++;
      }
    }
    
    // "normalize" coefficients (make everything use the preferred power)  
    for (auto & Factor : Factors) {
      if (Factor.Unit && Factor.Unit->PreferredPower != 0) {
	Factor.Coefficient = pow(10,Factor.Unit->PreferredPower);
	Coefficient /= pow(Factor.Coefficient,Factor.Power);
      }
    }
    // Resort according to power: 
    // Powers of 1.0 go first, followed by powers > 1.0 in increasing order, followed by powers < 1.0 in decreasing order
    std::sort(Factors.begin(),Factors.end(), [](const UnitFactor & FactorA, const UnitFactor & FactorB) {
					       // return true when a < b
					       if (FactorA.Power==1.0 && FactorB.Power != 1.0) {
						 return true;
					       }
					       if (FactorA.Power > 1.0 && FactorB.Power < 1.0) {
						 return true; 
					       }
					       if (FactorA.Power > 1.0 && FactorB.Power > 1.0) {
						 return FactorA.Power < FactorB.Power;
					       }
					       if (FactorA.Power < 1.0 && FactorB.Power < 1.0) {
						 return FactorA.Power > FactorB.Power;
					       }
					       return false;
					     });

    
    return std::make_tuple(Factors,Coefficient);
  }
  
  class UnitDB {
  public:
    std::unordered_map<std::string,std::shared_ptr<UnitDef>> UnitDict;
    std::unordered_map<std::string,std::shared_ptr<UnitDef>> MeasDict;
    
    UnitDB(std::vector<std::tuple<std::string,std::string,std::string,std::vector<std::string>,double,bool>> UnitDefs) // Unitdefs tuple consists of members of Class UnitDef, except for Index
    {
      std::string MeasName; // name of quantity measured by this unit
      std::string SingularName; // singular name of unit (e.g. meter)
      std::string PluralName; // plural name of unit (e.g. meters)
      std::vector<std::string> AbbrevList; // list of abbreviations
      double PreferredPower; // Preferred power of 10; ... typically 0, but 3 for grams indicating usual use of kh
      bool SiPrefixFlag;
      
      size_t Index=0; // index of creation of this unit
      
      for (auto & Def : UnitDefs) {
	std::tie(MeasName,SingularName,PluralName,AbbrevList,PreferredPower,SiPrefixFlag)=Def;
	
	std::shared_ptr<UnitDef> NewDef=std::make_shared<UnitDef>(MeasName,SingularName,PluralName,AbbrevList,Index,PreferredPower,SiPrefixFlag);
	UnitDict[SingularName]=NewDef;
	UnitDict[PluralName]=NewDef;
	MeasDict[MeasName]=NewDef;
	
	Index++;
      
      }
    }
  };
  
  static const UnitDB Basic_Units({
				// Parameters are MeasName, SingularName, PluralName, AbbrevList, PreferredPower, SiPrefixFlag
			    {"mass","gram","grams",{"g"},3,true},
			    {"number","unitless","unitless",{""},0,false},
			    {"number","percent","percent",{"%"},0,false},
			    {"energy","Joule","Joules",{"J"},0,true},
			    {"power","Watt","Watts",{"W"},0,true},
			    {"voltage","Volt","Volts",{"V"},0,true},
			    {"current","Amp","Amps",{"A"},0,true},
			    {"length","meter","meters",{"m"},0,true},
			    {"time","second","seconds",{"s"},0,true},
			    {"frequency","Hertz","Hertz",{"Hz"},0,true},
			    // include velocity here? Python version did...
			    {"angle","radian","radians",{"r"},0,true},
			    {"force","Newton","Newtons",{"N"},0,true},
			    {"force","pound","pounds",{"lb"},0,false},
			    {"force","ounce","ounces",{"oz"},0,false},
			    {"pressure","Pascal","Pascals",{"Pa"},0,true},
			    {"temperature","Kelvin","Kelvin",{"K"},0,true},
			    {"arbitrary","arbitrary","arbitrary",{"Arb"},0,true},
    });

  static const struct {
    std::string Prefix;
    std::string Abbrev;
    int Power;
  } SIPrefixes[] = {
		  {"Yotta","Y",24},
		  {"Zetta", "Z", 21},
		  {"Exa", "E", 18},
		  {"Peta", "P", 15},
		  {"Tera", "T", 12},
		  {"Giga", "G", 9},
		  {"Mega", "M", 6},
		  {"kilo", "k", 3},
		  {"milli", "m", -3},
		  {"micro", "u", -6}, 
		  {"nano", "n", -9}, 
		  {"pico", "p", -12},
		  {"femto", "f", -15}, 
		  {"atto", "a", -18},
		  {"zepto", "z", -21}, 
		  {"yocto", "y", -24},

  };
  
static inline  std::tuple<bool,std::string,int> HasSIPrefix(std::string Name)
  {
    //  look up unabbreviated prefixes, case insensitive
    for (auto & SIPrefix : SIPrefixes) {
      size_t cnt;
      for (cnt=0;cnt < SIPrefix.Prefix.size() && cnt < Name.size();cnt++) {
	if (std::toupper(Name[cnt] != std::toupper(SIPrefix.Prefix[cnt]))) {
	  break;
	}
      }
      if (cnt==SIPrefix.Prefix.size()) { // made it all the way through the previous loop.. a match!
	return std::make_tuple(true,SIPrefix.Prefix,SIPrefix.Power);
      }
    }

    if (Name.size() < 1) {
      return std::make_tuple(false,"",0);
    }
    // look up abbreviated prefixes, case sensitive
    for (auto & SIPrefix:  SIPrefixes) {
      if (Name[0]==SIPrefix.Abbrev[0]) {
	return std::make_tuple(true,SIPrefix.Abbrev,SIPrefix.Power);
      }
    }
    return std::make_tuple(false,"",0);

  }


  
  static const std::regex findpower_right("(.*?)(\\^)([-+]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[eE][-+]?\\d+)?)$");
  static const std::regex findfloat_right("(.*?)([-+]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[eE][-+]?\\d+)?)$");
  static const std::regex findunitname_right("(.*?)([a-zA-Z%]+)$");
  
  class units {
  public:

    std::vector<UnitFactor> Factors;
    double Coefficient;

    units() :
      Coefficient(1.0)
    {
      
    }

    units(std::vector<UnitFactor> Factors,double Coefficient) :
      Factors(Factors),
      Coefficient(Coefficient)
    {
      
    }

    friend units operator*(const units& lhs,const units& rhs) {
      std::vector<UnitFactor> allfactors=lhs.Factors;
      // Add rhs factors to allfactors (concatenating)
      allfactors.insert(allfactors.end(),rhs.Factors.begin(),rhs.Factors.end());
      
      return units(allfactors,lhs.Coefficient*rhs.Coefficient);
    }

    friend units operator*(const units& lhs,double rhs) {
      
      return units(lhs.Factors,lhs.Coefficient*rhs);
    }

    units& operator*=(const units& rhs) {
      *this = (*this)*rhs;
      return *this;
    }

    units& operator*=(double rhs) {
      *this = (*this)*rhs;
      return *this;
    }

    units power(double powerexp)
    {
      std::vector<UnitFactor> newfactors;
      
      for (auto & factor : Factors) {	
	newfactors.emplace_back(factor.Unit,factor.NameOnly,factor.Power*powerexp,factor.Coefficient);

      }
      return units(newfactors,pow(Coefficient,powerexp));
    }

    friend units operator/(const units& lhs,const units& rhs) {
      std::vector<UnitFactor> allfactors=lhs.Factors;
      // Add rhs factors to allfactors (concatenating)

      for (auto &factor : rhs.Factors) {
	allfactors.emplace_back(factor.Unit,factor.NameOnly,-factor.Power,factor.Coefficient);
	
      }
      allfactors.insert(allfactors.end(),rhs.Factors.begin(),rhs.Factors.end());

      double coefficient=1.0;
      std::tie(allfactors,coefficient)=sortunits(allfactors);
      
      return units(allfactors,coefficient*lhs.Coefficient/rhs.Coefficient);
    }

    friend units operator/(const units& lhs,double rhs) {
      
      return units(lhs.Factors,lhs.Coefficient/rhs);
    }

    units& operator/=(const units& rhs) {
      *this = (*this)/rhs;
      return *this;
    }

    units& operator/=(double rhs) {
      *this = (*this)/rhs;
      return *this;
    }

    units AddUnitFactor(std::string FactorName)
    // Note that unlike previous implementations, this returns a new units object 
    {
      std::vector<UnitFactor> allfactors=Factors;

      FactorName=stripstr(FactorName);
      double NewCoefficient=1.0;
      std::shared_ptr<UnitDef> GotUnit=nullptr;
      bool FoundSiPrefix=false;
      std::string SiPrefix;
      int SiPrefixPower;

      if (Basic_Units.UnitDict.find(FactorName) != Basic_Units.UnitDict.end()) {
	GotUnit=Basic_Units.UnitDict.at(FactorName);	
      }
      std::tie(FoundSiPrefix,SiPrefix,SiPrefixPower)=HasSIPrefix(FactorName);

      if (!GotUnit && FoundSiPrefix) {
	std::string TryFactorName=FactorName.substr(SiPrefix.size());
	if (Basic_Units.UnitDict.find(TryFactorName) != Basic_Units.UnitDict.end()) {
	  GotUnit=Basic_Units.UnitDict.at(TryFactorName);
	  NewCoefficient = pow(10.0,SiPrefixPower);
	}
      }

      if (GotUnit) {
	// Found Unit structure
	allfactors.emplace_back(GotUnit,"",1.0,NewCoefficient);
      } else {
	allfactors.emplace_back(nullptr,FactorName,1.0,NewCoefficient);
      }

      return units(allfactors,Coefficient);
    }


    units simplify() const
    {
      // Not fully implemented
      std::vector<UnitFactor> SortedFactors;
      double SortedCoeff;
      std::tie(SortedFactors,SortedCoeff) = sortunits(Factors);
      
      // Strip unitless, unless it is all there is: 
      if (SortedFactors.size() > 1) {
	size_t SortedFactorsInitialSize=SortedFactors.size();
	for (size_t FactorCnt=0; FactorCnt < SortedFactorsInitialSize;FactorCnt++) {
	  UnitFactor &Factor=SortedFactors[SortedFactorsInitialSize-FactorCnt-1];
	  if (Basic_Units.UnitDict.find("unitless") != Basic_Units.UnitDict.end() && Factor.Unit == Basic_Units.UnitDict.at("unitless")) {
	    SortedFactors.erase(SortedFactors.begin()+SortedFactorsInitialSize-FactorCnt-1);
	  }
	}
      }
      
      return units(SortedFactors,SortedCoeff*Coefficient);
      
    }
    
    static std::tuple<std::string,units> parseunitpower_right(std::string unitstr)
    {
      double power=1.0;
      std::smatch match;
      std::string remaining;
      units unitsobj;
      bool success=false;
      
      // parse ^real number on rhs
      if (std::regex_match(unitstr,match,findpower_right)) {
	power=strtod(std::string(match[3]).c_str(),NULL);
	unitstr=match[1];
      }

      if (unitstr.size() > 0 && unitstr[unitstr.size()-1]==')') {
	// ends-with(')')
	// parenthesis
	std::tie(remaining,unitsobj)=parseunits_right(unitstr.substr(0,unitstr.size()-1));

        if (remaining.size() < 1 || remaining[remaining.size()-1] != '(') {
	  throw std::runtime_error("Mismatched Parentheses in "+unitstr);
	}
	
        remaining=stripstr(remaining.substr(0,remaining.size()-1));
        return std::make_tuple(remaining,unitsobj);

      }
      
      // Try to match a coefficient
      if (std::regex_match(unitstr,match,findfloat_right)) {
	remaining=stripstr(match[1]);
	
        double coefficient=strtod(std::string(match[2]).c_str(),NULL);
        unitsobj=units();
	unitsobj *= coefficient;

	success=true;
      }

      // Try to match a unit name
      if (std::regex_match(unitstr,match,findunitname_right)) {
	remaining=stripstr(match[1]);
	std::string unitname=match[2];
        unitsobj=units();
        unitsobj=unitsobj.AddUnitFactor(unitname);

	success=true;
      }
      if (!success) {
        throw std::runtime_error("Parse error on R.H.S of " + unitstr);
      }

      if (power != 1.0) {
	unitsobj=unitsobj.power(power);
      }

      return std::make_tuple(remaining,unitsobj);
    }


    static std::tuple<std::string,units> parseunits_right(std::string unitstr)
    {
      std::string remaining,remaining2;
      units unitpower,unitsobj;

      std::tie(remaining,unitpower)=parseunitpower_right(unitstr);
      remaining=stripstr(remaining); 

      if (remaining.size() > 0 && remaining[remaining.size()-1]=='*') {
	// ends-with '*'
	std::tie(remaining2,unitsobj)=parseunits_right(stripstr(remaining.substr(0,remaining.size()-1)));
	return std::make_tuple(remaining2,unitsobj*unitpower);
      } else if (remaining.size() > 0 && remaining[remaining.size()-1]=='-') {
	// ends-with '-'
	std::tie(remaining2,unitsobj)=parseunits_right(stripstr(remaining.substr(0,remaining.size()-1)));
	return std::make_tuple(remaining2,unitsobj*unitpower);
      } else if (remaining.size() > 0 && remaining[remaining.size()-1]=='/') {
	// ends-with '/'
	std::tie(remaining2,unitsobj)=parseunits_right(stripstr(remaining.substr(0,remaining.size()-1)));
	return std::make_tuple(remaining2,unitsobj/unitpower);
      } else if (remaining.size() > 0 && (remaining[remaining.size()-1] > 32  || remaining[remaining.size()-1] < 0)) {
	// essentially ends-with(isalpha())
	std::tie(remaining2,unitsobj)=parseunits_right(stripstr(remaining.substr(0,remaining.size()-1)));
	return std::make_tuple(remaining2,unitsobj*unitpower);
	
      }

      return std::make_tuple(remaining,unitpower);
    }
    

    static units parseunits(std::string unitstr)
    {
      std::string unitstrstrip=stripstr(unitstr);

      if (unitstrstrip=="") {
	return units();
      }

      std::string remaining;
      units unitsobj;
      double coefficient;
      
      std::tie(remaining,unitsobj)=units::parseunits_right(unitstrstrip);
      assert(remaining.size()==0);

      std::vector<UnitFactor> Factors=unitsobj.Factors;
      std::tie(Factors,coefficient)=sortunits(Factors);
      return units(Factors,coefficient*unitsobj.Coefficient);      
    }

    static double comparerawunits(const units & CombA,const units &CombB)
    {
      // returns 0 for non-equal, When the unit combinations are equivalent, the coefficient of CombA relative
      // to CombB is returned. 
      // CombA and CombB MUST be already sorted with sortunits() method

      if (CombA.Factors.size() != CombB.Factors.size()) {
        return 0.0;
      }
      
      for (size_t Pos=0;Pos < CombA.Factors.size();Pos++) {
        const UnitFactor & FactorA=CombA.Factors[Pos];
	const UnitFactor & FactorB=CombB.Factors[Pos];
	
        if (FactorA != FactorB) {
	  return 0.0;
	} else {
	  if (FactorA.Power != FactorB.Power) {
	    return 0.0;
	  }
	  assert(FactorA.Coefficient==FactorB.Coefficient); // Should always match because of normalization in sorting function
	}
      }
      return CombA.Coefficient/CombB.Coefficient;
    }

    
    static bool compareunits(const units &comba, const units &combb)
    {
      return comparerawunits(comba.simplify(),combb.simplify());
    }
  };

  class Equivalence {
  public:
    std::shared_ptr<units> ToReplace;
    std::shared_ptr<units> ReplaceWith;

    Equivalence(std::shared_ptr<units> ToReplace, std::shared_ptr<units> ReplaceWith) :
      ToReplace(ToReplace),
      ReplaceWith(ReplaceWith)
    {
      
    }

    
  };

}

#endif // SNDE_UNITS_HPP
